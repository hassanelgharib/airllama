"""Test script to verify the API server starts without AirLLM dependencies."""

import sys
import time
import requests
from multiprocessing import Process

def start_server():
    """Start the uvicorn server."""
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=11434, log_level="info")

def test_endpoints():
    """Test basic endpoints."""
    base_url = "http://localhost:11434"
    
    # Wait for server to start
    print("Waiting for server to start...")
    for i in range(10):
        try:
            response = requests.get(f"{base_url}/")
            if response.status_code == 200:
                print("✓ Server started successfully")
                break
        except requests.ConnectionError:
            time.sleep(1)
    else:
        print("✗ Server failed to start")
        return False
    
    # Test root endpoint
    print("\n=== Testing Root Endpoint ===")
    try:
        response = requests.get(f"{base_url}/")
        print(f"GET / -> {response.status_code}")
        print(f"Response: {response.text}")
        assert response.status_code == 200
        assert "Ollama is running" in response.text
        print("✓ Root endpoint works")
    except Exception as e:
        print(f"✗ Root endpoint failed: {e}")
        return False
    
    # Test health endpoint
    print("\n=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"GET /health -> {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        print("✓ Health endpoint works")
    except Exception as e:
        print(f"✗ Health endpoint failed: {e}")
        return False
    
    # Test version endpoint
    print("\n=== Testing Version Endpoint ===")
    try:
        response = requests.get(f"{base_url}/api/version")
        print(f"GET /api/version -> {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        print("✓ Version endpoint works")
    except Exception as e:
        print(f"✗ Version endpoint failed: {e}")
        return False
    
    # Test list models endpoint
    print("\n=== Testing List Models Endpoint ===")
    try:
        response = requests.get(f"{base_url}/api/tags")
        print(f"GET /api/tags -> {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        print("✓ List models endpoint works")
    except Exception as e:
        print(f"✗ List models endpoint failed: {e}")
        return False
    
    # Test OpenAI models endpoint
    print("\n=== Testing OpenAI Models Endpoint ===")
    try:
        response = requests.get(f"{base_url}/v1/models")
        print(f"GET /v1/models -> {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        print("✓ OpenAI models endpoint works")
    except Exception as e:
        print(f"✗ OpenAI models endpoint failed: {e}")
        return False
    
    print("\n=== All Basic Tests Passed ===")
    return True

if __name__ == "__main__":
    print("Starting API server test...\n")
    
    # Start server in background
    server_process = Process(target=start_server)
    server_process.start()
    
    try:
        # Run tests
        success = test_endpoints()
        
        if success:
            print("\n✓✓✓ All tests passed! ✓✓✓")
            sys.exit(0)
        else:
            print("\n✗✗✗ Some tests failed ✗✗✗")
            sys.exit(1)
    finally:
        # Stop server
        print("\nStopping server...")
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()
        print("Server stopped")
