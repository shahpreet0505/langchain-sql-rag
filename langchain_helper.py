from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from few_shots import few_shots

import os
from dotenv import load_dotenv

load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
persist_dir = "fewshot_chroma"





def get_few_shot_db_chain():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=GROQ_KEY
    )

    db_user = "root"
    db_password = "your_database_password"
    db_host = "localhost"
    db_name = "amazon_tshirts"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                              sample_rows_in_table_info=3)

    schema = db.get_table_info()
    print("schema :", schema)

    fewshot_docs = []

    for ex in few_shots:
        text = f"""Q: {ex['Question']}
                A: {ex['SQLQuery']}"""

        fewshot_docs.append(
            Document(page_content=text, metadata={"type": "sql_example"})
        )

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(
        fewshot_docs, emb, persist_directory=persist_dir
    )

    vector_store.persist()

    vector_store = Chroma(
        embedding_function=emb,
        persist_directory=persist_dir
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    def get_fewshot_examples(query):
        docs = retriever.invoke(query)
        return "\n\n".join(d.page_content for d in docs)

    prompt = PromptTemplate.from_template("""
    You are a MySQL expert. Use the schema and examples to write a correct SQL query.
    Return ONLY SQL. No explanations. No ```.

    EXAMPLES:
    {examples}

    RULES:
    - If the question asks "how many" or "left in stock", use SUM(stock_quantity)
    - Always return a single SQL query
    - Use only valid columns

    Schema:
    {schema}

    Question:
    {question}

    SQL:
    """)

    sql_chain = prompt | llm | StrOutputParser()

    return sql_chain, db, schema, get_fewshot_examples








