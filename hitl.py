"""Human-in-the-loop module for travel assistant."""
from typing import Dict, Any, Optional, Union, List
import json
from datetime import datetime
from config import HITL_ENABLED, HITL_TIMEOUT

def get_human_approval(data: Dict[str, Any], context: str) -> bool:
    """Get human approval for a decision."""
    if not HITL_ENABLED:
        return True
    
    print("\n=== Human Approval Required ===")
    print(f"Context: {context}")
    print("Data to approve:")
    print(json.dumps(data, indent=2))
    
    response = input("\nDo you approve? (yes/no): ").lower().strip()
    return response in ['y', 'yes']

def get_human_selection(options: List[Dict[str, Any]], context: str) -> Optional[Dict[str, Any]]:
    """Get human selection from a list of options."""
    if not HITL_ENABLED:
        return options[0] if options else None
    
    print("\n=== Human Selection Required ===")
    print(f"Context: {context}")
    print("\nAvailable options:")
    
    for i, option in enumerate(options, 1):
        print(f"\n{i}. {json.dumps(option, indent=2)}")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (number): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def get_human_feedback(prompt: str) -> str:
    """Get free-form feedback from human."""
    if not HITL_ENABLED:
        return ""
    
    print("\n=== Human Feedback Required ===")
    print(prompt)
    return input("Your feedback: ").strip()

def validate_itinerary(itinerary: Dict[str, Any]) -> Dict[str, Any]:
    """Get human validation and modifications for an itinerary."""
    if not HITL_ENABLED:
        return itinerary
    
    print("\n=== Itinerary Validation Required ===")
    print("Current itinerary:")
    print(json.dumps(itinerary, indent=2))
    
    if not get_human_approval(itinerary, "Please review the itinerary"):
        feedback = get_human_feedback("What changes would you like to make?")
        itinerary['human_feedback'] = feedback
        itinerary['needs_revision'] = True
    else:
        itinerary['human_approved'] = True
        itinerary['approval_time'] = datetime.now().isoformat()
    
    return itinerary

def handle_error(error: Union[str, Dict[str, Any]], context: str) -> Dict[str, Any]:
    """Handle errors with human intervention."""
    if not HITL_ENABLED:
        return {"status": "error", "message": str(error)}
    
    print("\n=== Error Resolution Required ===")
    print(f"Context: {context}")
    print(f"Error: {error}")
    
    resolution = get_human_feedback("How would you like to resolve this error?")
    
    return {
        "status": "resolved" if resolution else "error",
        "original_error": error,
        "resolution": resolution,
        "timestamp": datetime.now().isoformat()
    } 