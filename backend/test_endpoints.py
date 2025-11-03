#!/usr/bin/env python3
"""
Simple test script to verify the new parallel image generation endpoints work.
"""

import requests
import json
import time

def test_endpoint(endpoint, descriptor):
    """Test a specific endpoint."""
    print(f"\n🧪 Testing {endpoint}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f'http://localhost:8000{endpoint}', 
            json={'descriptor': descriptor}, 
            timeout=60  # Longer timeout for image generation
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"Status: {response.status_code}")
        print(f"Time: {elapsed_time:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success!")
            print(f"Session ID: {data.get('session_id', 'N/A')}")
            print(f"Stage: {data.get('stage', 'N/A')}")
            print(f"Images: {len(data.get('images', []))}")
            
            # Show first few images
            images = data.get('images', [])
            for i, img in enumerate(images[:2]):  # Show first 2 images
                print(f"  Image {i+1}: {img.get('id', 'N/A')} -> {img.get('url', 'N/A')}")
            
            if len(images) > 2:
                print(f"  ... and {len(images) - 2} more images")
                
            return True, elapsed_time
            
        else:
            print(f"❌ Error: {response.text}")
            return False, elapsed_time
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - make sure the server is running on localhost:8000")
        return False, 0
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False, elapsed_time
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False, 0

def main():
    """Test both new endpoints."""
    print("🚀 Testing Parallel Image Generation Endpoints")
    print("=" * 60)
    
    descriptor = "A cozy living room with warm lighting"
    
    # Test results
    results = {}
    
    # Test fast sequential endpoint
    success1, time1 = test_endpoint('/api/generate-fast', descriptor)
    results['fast_sequential'] = {'success': success1, 'time': time1}
    
    # Test fast parallel endpoint  
    success2, time2 = test_endpoint('/api/generate-fast-parallel', descriptor)
    results['fast_parallel'] = {'success': success2, 'time': time2}
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    
    for endpoint, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        time_str = f"{result['time']:.2f}s" if result['time'] > 0 else "N/A"
        print(f"{endpoint:20} {status:8} {time_str}")
    
    if success1 and success2:
        print(f"\n🎉 Both endpoints working!")
        if time1 > 0 and time2 > 0:
            speedup = time1 / time2 if time2 > 0 else 0
            print(f"Speed comparison: Fast Parallel is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than Fast Sequential")
    else:
        print(f"\n⚠️  Some endpoints failed. Check server logs for details.")

if __name__ == "__main__":
    main()
