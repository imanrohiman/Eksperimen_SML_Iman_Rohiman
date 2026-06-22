# 3.prometheus_exporter.py
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])
ACTIVE_REQUESTS = Gauge('http_active_requests', 'Active HTTP requests')

def simulate_traffic():
    """Simulate HTTP traffic for monitoring"""
    methods = ['GET', 'POST', 'PUT', 'DELETE']
    endpoints = ['/api', '/health', '/predict', '/metrics']
    
    while True:
        method = random.choice(methods)
        endpoint = random.choice(endpoints)
        
        REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
        ACTIVE_REQUESTS.inc()
        
        # Simulate processing time
        latency = random.uniform(0.01, 0.5)
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)
        time.sleep(latency)
        
        ACTIVE_REQUESTS.dec()
        time.sleep(random.uniform(0.1, 1))

if __name__ == "__main__":
    print("🚀 Starting Prometheus Exporter...")
    
    # Start Prometheus server
    start_http_server(8001)
    logger.info("✅ Prometheus exporter running on port 8001")
    
    # Start simulation
    try:
        simulate_traffic()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
