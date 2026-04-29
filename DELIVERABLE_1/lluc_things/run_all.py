"""
run_all.py
List available skeletons and print next steps.
"""
import os

BASE = os.path.dirname(__file__)

if __name__ == "__main__":
    print("Submission skeletons:")
    for fname in sorted(os.listdir(BASE)):
        if fname.endswith('.py') and fname != 'run_all.py':
            print('-', fname)
    print("\nEdit each skeleton to implement the tasks requested in DELIVERABLE_1/enunciat.pdf and then run tests or create a report.")
