from llama_index.llms.ollama import Ollama

#from llama_parse import LlamaParse
from llama_index.readers.file import PyMuPDFReader

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
#from llama_index.core.output_parsers import PydanticOutputParser
#from llama_index.core.query_pipeline import QueryPipeline

from prompts import context


# Get the Ollama LLM instance

llm = Ollama(model="phi3:mini", request_timeout=120.0)



parser = PyMuPDFReader()                                           
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

#parser = LlamaParse(result_type="markdown")
# file_extractor = {".pdf": parser}
# documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

#this model is cached somewhere else other than this workspace, fix that pathway 
# #from sentence-transformers environment later
embed_model = resolve_embed_model("local:dbmdz/bert-base-turkish-uncased")
"""
from llama_index.embeddings import HuggingFaceEmbedding

embedding = HuggingFaceEmbedding(
    model_name='bert-base-uncased',
    tokenizer_name='bert-base-uncased',
    cache_folder='/path/to/your/local/model'
)"""
#embed_model = resolve_embed_model("local:BAAI/bge-m3")

vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
#The simplest way to store your indexed data is to use the built-in .persist()
#  method of every Index, which writes all the data to disk at the location specified. 
# This works for any type of index.
#vector_index.persist("./vector_index")
# rebuild storage context
#storage_context = StorageContext.from_defaults(persist_dir="<persist_dir>")

# load index
#index = load_index_from_storage(storage_context)

query_engine = vector_index.as_query_engine(llm=llm)

tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="legal_contract",
            description="This gives documentation about Legal contracts. Use it for reading legal contracts and interacting with them.",
        ),
    ),
]

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

"""
# build index
index = VectorStoreIndex.from_documents(documents)

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

# query
response = query_engine.query("What did the author do growing up?")
print(response)"""

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0

    while retries < 3:
        try:
            result = agent.query(prompt)
            print(result)   
            break
        except Exception as e:
            retries += 1
            print(f"Error occured, retry #{retries}:", e)

    if retries >= 3:
        print("Unable to process request, try again...")
        continue

