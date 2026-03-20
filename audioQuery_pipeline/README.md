# Query Audio Pipeline Demo

Set environment to where SenseVoice is installed
```
export PYTHONPATH=~/workspace/vortex/audioQuery_pipeline/python_udls/SenseVoice:$PYTHONPATH
```

Note: In paper, used FlagEmbedding as dependencies for encode_udl. However, this libary has dependencies issue with Pipeline1, so we replaced that with SentenceTransformer library instead.