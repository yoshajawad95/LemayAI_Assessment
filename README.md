Architecture Overview:

FastAPI server with async support for handling multiple requests
Uvicorn workers for production-grade concurrency
NGINX load balancer for request distribution
Docker containers with proper caching and scaling

Loads DistilBERT from HuggingFace
Parallel request processing - 10 concurrent sentiment analysis requests
High performance - Sub-second processing of multiple requests
Production architecture - NGINX + uvicorn + FastAPI

Why I chose DistilBERT for sentiment analysis as the demo:

Fast & Lightweight - DistilBERT is 60% smaller than BERT but retains 97% performance
Pre-trained & Ready - No additional fine-tuning needed
Clear Output - Sentiment analysis gives easily interpretable results
Low Memory - Won't overwhelm systems during parallel testing
Popular Use Case - Sentiment analysis is a common real-world application

The TestNB notebook shows how to make parallel requests and compares performance between sequential vs parallel processing. 
It tests sentiment analysis on multiple texts simultaneously and includes benchmarking to demonstrate the server's concurrent capabilities.

Request Flow

Notebook sends HTTP POST to http://localhost/predict
NGINX receives the request and forwards it to one of the FastAPI servers
FastAPI server checks if the model is cached, loads it if not
HuggingFace pipeline runs inference on the input text
Response travels back through NGINX to your notebook

Parallel Processing
The notebook uses ThreadPoolExecutor to send multiple requests simultaneously:

Sequential: Sends one request, waits for response, then sends next
Parallel: Sends all requests at once, collects responses as they complete



