#!/usr/bin/env python3
"""
Test script for the database conversation history functionality.
"""

import sys
import os

# Add the current directory to the Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.database import store_message, get_conversation_history, get_conversation_messages

def test_conversation_history():
    """Test the conversation history functionality."""
    print("Testing conversation history functionality...")
    
    # Test conversation ID
    test_conversation_id = "test_conv_123"
    
    # Clear any existing test data (optional - for clean testing)
    print(f"Testing with conversation ID: {test_conversation_id}")
    
    # Test storing messages
    print("\n1. Testing message storage...")
    
    # Store user message
    result1 = store_message(test_conversation_id, "user", "Hello, I'm looking for RC cars")
    print(f"Stored user message: {result1}")
    
    # Store bot response
    result2 = store_message(test_conversation_id, "bot", "Hi! I'd be happy to help you find RC cars. What type are you interested in?")
    print(f"Stored bot message: {result2}")
    
    # Store another user message
    result3 = store_message(test_conversation_id, "user", "I want something for beginners under $200")
    print(f"Stored second user message: {result3}")
    
    # Store another bot response
    result4 = store_message(test_conversation_id, "bot", "Great! Here are some beginner-friendly RC cars under $200...")
    print(f"Stored second bot message: {result4}")
    
    # Test retrieving conversation history
    print("\n2. Testing conversation history retrieval...")
    history = get_conversation_history(test_conversation_id)
    print(f"Retrieved history: {history}")
    
    # Test retrieving detailed messages
    print("\n3. Testing detailed message retrieval...")
    messages = get_conversation_messages(test_conversation_id)
    print("Retrieved messages:")
    for i, (role, message, timestamp) in enumerate(messages, 1):
        print(f"  {i}. [{role}] {timestamp}: {message}")
    
    # Test with non-existent conversation
    print("\n4. Testing with non-existent conversation...")
    empty_history = get_conversation_history("non_existent_conv")
    print(f"Empty conversation history: '{empty_history}'")
    
    # Test error handling
    print("\n5. Testing error handling...")
    try:
        invalid_result = store_message(test_conversation_id, "invalid_role", "This should fail")
        print(f"Invalid role test result: {invalid_result}")
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    try:
        test_conversation_history()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
