python 3.11
pip install -r req.txt

1. use `init_db.py`to create vectorDB based on files in `raw_docs` folder (put your documents in this folder or change the path in `init_db.py`)
2. run `web.py` to start the web ui
3. modify `model.py` about calling LLM
4. The page numbers corresponding to the first four questions in `qa.json` are the pages of the pdf file, and the page numbers marked for the remaining questions are the pages of the file content.
