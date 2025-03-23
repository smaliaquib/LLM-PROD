#!/usr/bin/env python3
import requests
import json
import os
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Load test for TorchServe model endpoint')
parser.add_argument('--concurrent', type=int, default=20, help='Number of concurrent requests')
parser.add_argument('--total', type=int, default=200, help='Total number of requests')
args = parser.parse_args()

# Get the environment variables
SERVICE_HOSTNAME = os.environ.get('SERVICE_HOSTNAME', 'torchserve.default.svc.cluster.local')
INGRESS_HOST = os.environ.get('INGRESS_HOST', '35.192.38.81')
INGRESS_PORT = os.environ.get('INGRESS_PORT', '80')

# Set up the URL
# URL = f"http://{INGRESS_HOST}:{INGRESS_PORT}/v1/models/mnist:predict"
URL = f"http://{INGRESS_HOST}:{INGRESS_PORT}/v1/models/prompt:predict"

HEADERS = {
    "Host": SERVICE_HOSTNAME,
    "Content-Type": "application/json"
}

# Load the mnist01.json file
with open('userprompt.json', 'r') as f:
    DATA = json.load(f)

# Synchronous version using ThreadPoolExecutor
def send_request():
    try:
        response = requests.post(URL, headers=HEADERS, json=DATA)
        return response.status_code
    except Exception as e:
        print(f"Error: {e}")
        return 0

def run_sync_load_test(concurrent, total_requests):
    print(f"Starting synchronous load test with {concurrent} concurrent requests, {total_requests} total requests")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        status_codes = list(executor.map(send_request, range(total_requests)))
    
    end_time = time.time()
    successful = status_codes.count(200)
    
    print(f"\nLoad test completed in {end_time - start_time:.2f} seconds")
    print(f"Requests per second: {total_requests / (end_time - start_time):.2f}")
    print(f"Successful requests: {successful} ({successful/total_requests*100:.2f}%)")
    print(f"Failed requests: {total_requests - successful}")

# Asynchronous version using aiohttp
async def async_send_request(session, request_id):
    try:
        start_time = time.time()
        async with session.post(URL, headers=HEADERS, json=DATA) as response:
            await response.text()
            status_code = response.status
            elapsed = time.time() - start_time
            if request_id % 20 == 0:
                print(f"Request {request_id} completed - Status: {status_code}, Time: {elapsed:.4f}s")
            return status_code
    except Exception as e:
        print(f"Error in request {request_id}: {e}")
        return 0

async def run_async_load_test(concurrent, total_requests):
    print(f"Starting asynchronous load test with {concurrent} concurrent requests, {total_requests} total requests")
    start_time = time.time()
    
    connector = aiohttp.TCPConnector(limit=concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(total_requests):
            tasks.append(async_send_request(session, i+1))
        
        # Process in batches to control concurrency
        status_codes = []
        for i in range(0, total_requests, concurrent):
            batch = tasks[i:min(i + concurrent, total_requests)]
            batch_results = await asyncio.gather(*batch)
            status_codes.extend(batch_results)
    
    end_time = time.time()
    successful = status_codes.count(200)
    
    print(f"\nLoad test completed in {end_time - start_time:.2f} seconds")
    print(f"Requests per second: {total_requests / (end_time - start_time):.2f}")
    print(f"Successful requests: {successful} ({successful/total_requests*100:.2f}%)")
    print(f"Failed requests: {total_requests - successful}")

if __name__ == "__main__":
    # Choose which version to run (async is generally more efficient)
    # run_sync_load_test(args.concurrent, args.total)
    asyncio.run(run_async_load_test(args.concurrent, args.total))
