#!/bin/bash
# Build SDTM RAG Knowledge Base

echo "Building SDTM RAG Knowledge Base..."

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements_rag.txt

# Build the knowledge base
echo "Building structured knowledge base with annotation instructions..."
python sdtm_rag_builder.py --base-path . --output-dir sdtm_rag_kb

echo "Knowledge base build complete!"
echo "Output saved to: sdtm_rag_kb/"

# Show statistics
echo ""
echo "Knowledge base statistics:"
if [ -f "sdtm_rag_kb/sdtm_rag_chunks.json" ]; then
    chunk_count=$(python -c "import json; print(len(json.load(open('sdtm_rag_kb/sdtm_rag_chunks.json'))))")
    echo "- Total RAG chunks: $chunk_count"
fi

# Test the retriever
echo ""
echo "Testing retriever with sample queries..."
echo "1. Testing CRF term mapping:"
python sdtm_rag_retriever.py --kb-dir sdtm_rag_kb --query "Date of Birth"

echo ""
echo "2. Testing annotation instruction retrieval:"
python sdtm_rag_retriever.py --kb-dir sdtm_rag_kb --query "supplemental qualifiers Other specify"

echo ""
echo "3. Testing when/then pattern:"
python sdtm_rag_retriever.py --kb-dir sdtm_rag_kb --query "temperature celsius findings domain"