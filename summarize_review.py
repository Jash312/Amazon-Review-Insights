import pymongo
from bson import ObjectId
import pandas as pd
from functools import lru_cache
from langchain_core.language_models import BaseLLM
import os
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
from tqdm.auto import tqdm
import sys

mongo_uri = "mongodb+srv://Admin:Admin1234@cluster0.lhuhlns.mongodb.net"

olama_config = {
    "model_name" : "mistral",
    "base_url": "https://funny-kiwi-visually.ngrok-free.app"
#     "base_url": "http://localhost:11434"

}

@lru_cache
def get_client(mongo_uri):
    client = pymongo.MongoClient(mongo_uri)
    return client

@lru_cache
def get_reviews(object_id):
    client = get_client(mongo_uri)
    
    db = client["Full_Stack_Project"]
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
                      organization=llm_config.get("organization"),
                      model_name=llm_config.get("model_name"),
                      streaming=False)

    return Ollama(model=llm_config.get("model_name"), base_url=llm_config.get("base_url"))


def summarize_by_feature(llm, product, feature, reviews):
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
    result = chain({"feature": feature, "product": product, "input_documents": reviews}, return_only_outputs=True)
    return result["output_text"]

def summarize(object_id):
    product_details, reviews = get_reviews(object_id)
    grouped_reviews_by_feature = group_by_features(reviews)
    llm = get_model(False, olama_config)
    summary_dict = {}
    for _, row in tqdm(grouped_reviews_by_feature.iterrows()):
        
        feature = row["features"]
        
        print("Processing - ", feature)
        
        chunked_reviews = split_text(row["concatenated"], count=1000)
        summary = summarize_by_feature(llm, product_details["Title"], feature, chunked_reviews)
        summary_dict[feature] = summary
        
    return summary_dict


if __name__ == "__main__":
    summaries = summarize(sys.argv[1])
    print(summaries)
