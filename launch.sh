# start streamlit app

# streamlit run app.py

# streamlit run app.py --server.address=0.0.0.0 --server.port=8501

mkdir -p logs

streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port=7860 \
  --logger.level=debug 
