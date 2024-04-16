import asyncio
import json
import os
import sys
import time
from functools import lru_cache

import pandas as pd
import pymongo
from bson import ObjectId
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain, StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_community.llms import Ollama
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

mongo_uri = "mongodb+srv://Admin:Admin1234@cluster0.lhuhlns.mongodb.net"

olama_config = {
    "model_name": "mistral",
    "base_url": "https://funny-kiwi-visually.ngrok-free.app"
}

openai_config = {
    "model_name": "gpt-3.5-turbo",
    "api_key": "",
}


@lru_cache
def get_db(mongo_uri):
    client = pymongo.MongoClient(mongo_uri)
    db = client["Full_Stack_Project"]
    return db


@lru_cache
def get_reviews(object_id):
    db = get_db(mongo_uri)

    collection = db["Amazon_Reviews"]

    id_condition = {"_id": ObjectId(object_id)}
    result = collection.find_one(id_condition)

    return result["Product_Details"], pd.DataFrame(result["Reviews"])


def group_by_features(df):
    df = df.explode('features')
    return df.groupby('features').apply(lambda x: '\n'.join(x['title'] + ': ' + x['review'])).reset_index(
        name='concatenated')


def split_text(text, count=1000):
    chunks = []
    current_chunk = ""
    lines = text.split("\n")

    for line in lines:
        if len(current_chunk) + len(line) > count:
            chunks.append(Document(page_content=current_chunk))
            current_chunk = ""
        current_chunk += line + "\n"

    if current_chunk:
        chunks.append(Document(page_content=current_chunk))

    return chunks


def get_model(is_production: bool, llm_config) -> BaseLLM:
    if is_production:
        os.environ["OPENAI_API_KEY"] = llm_config.get("api_key")
        return ChatOpenAI(temperature=0,
                          model_name=llm_config.get("model_name"),
                          streaming=False)

    return Ollama(model=llm_config.get("model_name"), base_url=llm_config.get("base_url"))


async def summarize_by_feature_refine_async(llm, product, feature, reviews):
    formatting_instructions = "{\"pros\": [\"Pros of the feature\"], \"cons\": [\"Cons of the feature\"] }"
    prompt_template = """Form the pros and cons of {product} only in terms of {feature} using the below reviews:
    Reviews are in the format "title: review"
    {text}
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final good and bad points\n"
        "We have provided an existing summary of product reviews up to a certain point: {existing_answer}\n"
        "We have the opportunity to extract the good and bad points about the product only in terms of {feature}"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original good and bad points only in terms of {feature} in English"
        "If the context isn't useful, return the original good and bad points. Maximum 5 points"
        "Formatting Instructions: {formatting_instructions}"
        "Output MUST be a JSON and should ADHERE TO FORMATTING INSTRUCTIONS."
    )
    refine_prompt = PromptTemplate.from_template(refine_template)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
    )
    result = chain({"feature": feature, "product": product, "input_documents": reviews,
                    "formatting_instructions": formatting_instructions}, return_only_outputs=True)
    return {feature: result["output_text"]}


async def summarize_by_feature_map_reduce_async(llm, product, feature, reviews):
    formatting_instructions = "{\"pros\": [\"Pros of the feature\"], \"cons\": [\"Cons of the feature\"] }"

    # Map
    map_template = """Form pros and cons for the {product} only in terms of {feature} feature using the below reviews:
        Reviews are in the format "title: review"
        {docs}
        If there is no pros and cons in terms of {feature} feature, return empty.
        """
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce
    reduce_template = """The following is set of pros and cons for the {product} in terms of {feature} feature:
    {docs}
    Take these and distill it into a final, consolidated pros and cons ONLY I N TERMS of {feature} feature. 

    Formatting Instructions: {formatting_instructions}
    Output MUST be a JSON and should ADHERE TO FORMATTING INSTRUCTIONS.
    """

    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    result = map_reduce_chain({"feature": feature, "product": product, "input_documents": reviews,
                               "formatting_instructions": formatting_instructions})
    return {feature: result["output_text"]}


async def summarize_by_features_async(llm, object_id):
    product_details, reviews = get_reviews(object_id)
    grouped_reviews_by_feature = group_by_features(reviews)

    tasks = []
    for _, row in grouped_reviews_by_feature.iterrows():
        feature = row["features"]

        print("Processing - ", feature)

        chunked_reviews = split_text(row["concatenated"], count=1000)
        tasks.append(summarize_by_feature_refine_async(llm, product_details["Title"], feature, chunked_reviews))

    results = await asyncio.gather(*tasks)

    print(results)

    summary_dict = {}

    for feature_dict in results:
        item = list(feature_dict.items())[0]

        feature = item[0]
        summary = item[1]
        summary_dict[feature] = json.loads(summary)

    return summary_dict, product_details


def update_summary(review_object_id, feature_summary, action_items):
    collection = get_db(mongo_uri)["Amazon_Reviews"]
    feature_data = {
        "review_id": review_object_id,
        "feature_summary": feature_summary
    }

    # Update one document matching the filter
    result = collection.update_one({"_id": ObjectId(review_object_id)},
                                   {"$set": {"Summary": feature_data, "ActionItems": action_items}})
    return result


def get_pros_cons_doc(features):
    feature_string = ""
    for feature in features:
        feature_string = f"{feature_string}\n\nFeature - {feature}:"

        pros = " ".join(features.get(feature).get("pros"))
        feature_string = f"{feature_string}\nPros:\n{pros}"

        cons = " ".join(features.get(feature).get("cons"))
        feature_string = f"{feature_string}\nCons:\n{cons}"

    return Document(page_content=feature_string)


def get_action_items(llm, product, pros_cons_by_features):
    pros_cons_doc = get_pros_cons_doc(pros_cons_by_features)

    prompt_template = """Provide the action items / recommendations for the product owner using below pros and cons 
    that are extracted from the customer reviews:
    Product - {product}
    Pros and cons - "{text}"

    Output FORMAT MUST ONLY be PIPE SEPARATED VALUES(|), Example: "action item 1|action item 2|action item n"."""
    prompt = PromptTemplate.from_template(prompt_template)

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    result = stuff_chain.run({'input_documents': [pros_cons_doc], 'product': product})

    split = result.split("|")

    if len(split) == 1:
        print("Output format is not right, splitting with new line", split)
        split = result.split("\n")

    return split


def main(review_id, is_prod, openai_key):
    start_time = time.time_ns()

    if is_prod:
        openai_config["api_key"] = openai_key
        llm = get_model(is_prod, openai_config)
    else:
        llm = get_model(is_prod, olama_config)

    pros_cons_by_features, product_details = asyncio.run(summarize_by_features_async(llm, review_id))
    action_items = get_action_items(llm, product_details["Title"], pros_cons_by_features)
    end_time = time.time_ns()

    print("Summary and Action Items Processing Time - ", end_time - start_time)

    update_summary(review_id, pros_cons_by_features, action_items)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] == "True", sys.argv[3])
