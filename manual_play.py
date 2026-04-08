from triage_env import MedicalTriageEnv, TriageAction

def main():
    print("🏥 Welcome to the Interactive Medical Triage Test 🏥")
    print("Choose task difficulty:")
    print("  (1) classify_urgency (Easy: Just pick urgency)")
    print("  (2) recommend_action (Medium: Urgency + Action)")
    print("  (3) evolving_symptoms (Hard: Multi-step trajectory)")
    choice = input("Enter 1, 2, or 3 [default=2]: ").strip()
    
    task_map = {"1": "classify_urgency", "2": "recommend_action", "3": "evolving_symptoms"}
    task_name = task_map.get(choice, "recommend_action")
    
    env = MedicalTriageEnv(task_name=task_name)
    obs = env.reset()
    
    done = False
    
    while not done:
        print("\n" + "="*55)
        print(f"Step: {obs.step_number}")
        print(f"Patient ID: {obs.patient_id} | Age: {obs.age} | Gender: {obs.gender}")
        print(f"History: {obs.medical_history}")
        print(f"Vitals: {obs.vital_signs}")
        print("-"*55)
        print(f"🚨 SYMPTOMS: {obs.current_symptoms}")
        print("-"*55)
        
        print("\nTask: " + obs.task_instructions)
        u = input("Assess Urgency (low / medium / high) [default=medium]: ").strip().lower()
        if u not in ["low", "medium", "high"]: 
            u = "medium"
            
        a = "clinic_visit"
        if task_name != "classify_urgency":
            a = input("Recommend Action (self_care / clinic_visit / emergency_room) [default=clinic_visit]: ").strip().lower()
            if a not in ["self_care", "clinic_visit", "emergency_room"]: 
                a = "clinic_visit"

        action = TriageAction(urgency_level=u, recommended_action=a, reasoning="Human manual input")
        
        obs, reward, done, info = env.step(action)
        
        print("\n>>> RESULT <<<")
        print(f"Reward Score: {reward.score:.2f} / 1.00")
        print(f"Feedback: {reward.feedback}")
        
        if done:
            print("-"*55)
            print(f"Ground Truth Urgency: {info.get('true_urgency')}")
            print(f"Ground Truth Action: {info.get('true_action')}")

    print("\nEpisode finished! Try running the script again to see a new randomized patient case.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting interactive play.")
