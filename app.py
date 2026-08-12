import streamlit as st
import uuid
from datetime import datetime, timezone
from supabase import create_client

st.title("GAIA Supabase Test")

supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"],
)

uploaded = st.file_uploader(
    "Choose a test image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    if st.button("Test Supabase Upload"):

        try:
            now = datetime.now(timezone.utc)
            file_id = str(uuid.uuid4())

            storage_path = (
                f"test/{file_id}.jpg"
            )

            image_bytes = uploaded.getvalue()

            # Upload image
            supabase.storage \
                .from_("gaia-images") \
                .upload(
                    storage_path,
                    image_bytes,
                    {
                        "content-type": uploaded.type,
                        "upsert": False,
                    },
                )

            # Insert database record
            result = (
                supabase
                .table("GAIA Diagnosis Database")
                .insert({
                    "image_path": storage_path,
                    "crop": "tomato",
                    "prediction": "TEST",
                    "confidence": 1.0,
                    "created_at": now.isoformat(),
                    "needs_human_review": False,
                    "human_diagnosis": None,
                    "reviewed_at": None,
                    "approved_for_training": False,
                })
                .execute()
            )

            st.success(
                "✅ Supabase test successful!"
            )

            st.write(result.data)

        except Exception as e:

            st.error(
                "❌ Supabase test failed"
            )

            st.code(str(e))
