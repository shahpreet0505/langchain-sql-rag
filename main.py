import streamlit as st
from langchain_helper import get_few_shot_db_chain

st.title("Amazon T Shirts: Database Q&A 👕")


sql_chain, db, schema, get_fewshot_examples = get_few_shot_db_chain()

question = st.text_input("Ask a question about inventory:")

if question:
    st.write("🔎 Retrieving few-shot examples...")
    examples = get_fewshot_examples(question)

    st.write("🧠 Generating SQL query...")
    sql = sql_chain.invoke({
        "schema": schema,
        "question": question,
        "examples": examples
    })

    st.code(sql, language="sql")

    st.write("📊 Running query...")
    result = db.run(sql)

    st.success("Answer:")
    st.write(result)