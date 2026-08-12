import streamlit as st
from supabase import create_client

st.title("GAIA Supabase Connection Test")

try:
    supabase = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )

    # Test the database connection
    response = (
        supabase
        .table("GAIA_Diagnosis")
        .select("id")
        .limit(1)
        .execute()
    )

    st.success("✅ Supabase connection is working.")
    st.write(response.data)

except Exception as e:
    st.error("❌ Supabase connection failed.")
    st.code(str(e))
