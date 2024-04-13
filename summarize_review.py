import pymongo
from bson import ObjectId
import pandas as pd
from functools import lru_cache
from langchain_core.language_models import BaseLLM
import os
import json
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
from tqdm import tqdm
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain, StuffDocumentsChain


mongo_uri = "mongodb+srv://Admin:Admin1234@cluster0.lhuhlns.mongodb.net"

olama_config = {
    "model_name" : "mistral",
    "base_url": "https://funny-kiwi-visually.ngrok-free.app"
}

openai_config = {
    "model_name" : "gpt-3.5-turbo",
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
    return df.groupby('features').apply(lambda x: '\n'.join(x['title'] + ': ' + x['review'])).reset_index(name='concatenated')


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


def summarize_by_feature_refine(llm, product, feature, reviews):
    formatting_instructions = "{\"pros\": [\"Pros of the feature\"], \"cons\": [\"Cons of the feature\"] }"
    prompt_template = """Summarize the feature of {product} only in terms of {feature} using the below reviews:
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
        "If the context isn't useful, return the original good and bad points."
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
    result = chain({"feature": feature, "product": product, "input_documents": reviews, "formatting_instructions":formatting_instructions}, return_only_outputs=True)
    return result["output_text"]


def summarize_by_feature_map_reduce(llm, product, feature, reviews):

    formatting_instructions = "{\"pros\": [\"Pros of the feature\"], \"cons\": [\"Cons of the feature\"] }"

    # Map
    map_template = """Extract pros and cons for the {product} only in terms of {feature} feature using the below reviews:
        Reviews are in the format "title: review"
        {docs}
        If there is no pros and cons interms of {feature} feature, return empty.
        Pros and Cons:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce
    reduce_template = """The following is set of pros and cons for the {product} in terms of {feature} feature:
    {docs}
    Take these and distill it into a final, consolidated pros and cons ONLY IN TERMS of {feature} feature. 

    Formatting Instructions: {formatting_instructions}
    "Output MUST be a JSON and should ADHERE TO FORMATTING INSTRUCTIONS.
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

    result = map_reduce_chain({"feature": feature, "product": product, "input_documents": reviews, "formatting_instructions":formatting_instructions})
    return result["output_text"]


def update_summary(review_object_id, feature_summary):
    collection = get_db["Feature_Summary"]
    feature_data = {
        "review_id": review_object_id,
        "feature_summary": feature_summary
    }
    
    return collection.insert_one(feature_data)

        
def summarize(object_id, is_prod, openai_key):
    product_details, reviews = get_reviews(object_id)
    grouped_reviews_by_feature = group_by_features(reviews)
    
    if is_prod:
        openai_config["api_key"] = openai_key
        llm = get_model(is_prod, openai_config)
    else:  
        llm = get_model(is_prod, olama_config)

    summary_dict = {}
    for _, row in tqdm(grouped_reviews_by_feature.iterrows()):
        
        feature = row["features"]
        
        print("Processing - ", feature)
        
        chunked_reviews = split_text(row["concatenated"], count=1000)
        summary = summarize_by_feature_refine(llm, product_details["Title"], feature, chunked_reviews)
        summary_dict[feature] = summary
        
    return summary_dict


def update_summary(review_object_id, feature_summary):
    collection = get_db(mongo_uri)["Feature_Summary"]
    feature_data = {
        "review_id": review_object_id,
        "feature_summary": feature_summary
    }
    
    result = collection.insert_one(feature_data)
    return str(res.inserted_id)


def main(review_id, is_prod, openai_key):
    summaries = summarize(review_id, is_prod, openai_key)
    update_summary(review_id, summaries)

    
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] == "True", sys.argv[3])
