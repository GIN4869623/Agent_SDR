# Scripts

## Import a PDF into the local knowledge base

The current RAG implementation only loads `.md` and `.txt` files from `data/`.
Use the importer below to convert a PDF into markdown chunks first:

```bash
python scripts/import_pdf_to_kb.py "C:\Users\21849\Desktop\USRP2943R手册.pdf" --doc-name "USRP2943R手册"
```

Each chunk is written into `data/<document-name>/` and will be loaded after the app restarts.
