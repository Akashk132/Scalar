document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const btnInit = document.getElementById('btnInit');
    const taskSelect = document.getElementById('taskSelect');
    const dashboard = document.getElementById('dashboard');
    const btnSubmit = document.getElementById('btnSubmit');
    const btnNext = document.getElementById('btnNext');
    const resultCard = document.getElementById('resultCard');
    const actionFormGroup = document.getElementById('actionFormGroup');
    const investigationCard = document.getElementById('investigationCard');
    const btnInvVitals = document.getElementById('btnInvVitals');
    const btnInvSymptoms = document.getElementById('btnInvSymptoms');
    
    let currentTask = "recommend_action";
    let isDone = false;
    
    // Toggle Group Logic
    const setupToggleGroups = () => {
        document.querySelectorAll('.btn-group').forEach(group => {
            group.addEventListener('click', (e) => {
                if (e.target.classList.contains('toggle-btn')) {
                    group.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('selected'));
                    e.target.classList.add('selected');
                    checkFormReady();
                }
            });
        });
    };

    const checkFormReady = () => {
        const u = document.querySelector('#urgencyGroup .selected');
        const a = document.querySelector('#actionGroup .selected');
        let ready = !!(u && a);
        if (ready) btnSubmit.classList.remove('disabled');
        else btnSubmit.classList.add('disabled');
    };

    // Reset API Call
    btnInit.addEventListener('click', async () => {
        currentTask = taskSelect.value;
        const res = await fetch(`/reset?task_name=${currentTask}`, { method: 'POST' });
        const obs = await res.json();
        
        // Hide result, show dashboard
        resultCard.classList.add('hidden');
        dashboard.classList.remove('hidden');
        
        // Reset toggles
        document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('selected'));
        checkFormReady();
        
        // Configure UI based on task
        if (currentTask === "direct_triage") {
            investigationCard.classList.add('hidden');
        } else {
            investigationCard.classList.remove('hidden');
            btnInvVitals.classList.remove('disabled');
            btnInvSymptoms.classList.remove('disabled');
        }

        populateObservation(obs);
    });

    const populateObservation = (obs) => {
        document.getElementById('patientId').innerText = obs.patient_id;
        document.getElementById('pAge').innerText = obs.age;
        document.getElementById('pGender').innerText = obs.gender;
        document.getElementById('pHistory').innerText = obs.medical_history;
        document.getElementById('pVitals').innerText = obs.discovered_vitals;
        document.getElementById('pSymptoms').innerText = obs.chief_complaint + " | Discovered: " + obs.discovered_symptoms;
        document.getElementById('stepCounter').innerText = obs.step_number;
        document.getElementById('taskInstructions').innerText = obs.task_instructions;
        
        // Add subtle animation to signify update
        const sympCard = document.querySelector('.symptoms-card');
        const vitalsCard = document.querySelector('.vitals-card');
        sympCard.style.transform = 'scale(1.02)';
        vitalsCard.style.transform = 'scale(1.02)';
        setTimeout(() => {
            sympCard.style.transform = 'scale(1)';
            vitalsCard.style.transform = 'scale(1)';
        }, 200);
    };

    // Investigation API Calls
    const doInvestigate = async (target) => {
        if (target === 'vitals') btnInvVitals.classList.add('disabled');
        if (target === 'symptoms') btnInvSymptoms.classList.add('disabled');
        
        const payload = {
            action_type: "investigate",
            investigation_target: target,
            reasoning: "Selected via Web UI"
        };

        const res = await fetch('/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        
        populateObservation(data.observation);
        
        // Show investigation cost penalty briefly
        const circle = document.querySelector('.circular-progress');
        document.getElementById('scoreValue').innerText = data.reward.score.toFixed(2);
        resultCard.classList.remove('hidden');
        document.getElementById('feedbackText').innerText = data.reward.feedback;
        document.getElementById('btnNext').classList.add('hidden');
        document.getElementById('truthBox').classList.add('hidden');
    };

    btnInvVitals.addEventListener('click', () => {
        if (!btnInvVitals.classList.contains('disabled')) doInvestigate('vitals');
    });

    btnInvSymptoms.addEventListener('click', () => {
        if (!btnInvSymptoms.classList.contains('disabled')) doInvestigate('symptoms');
    });

    // Step API Call
    btnSubmit.addEventListener('click', async () => {
        btnSubmit.classList.add('disabled');
        btnSubmit.innerText = "Processing...";

        const uBtn = document.querySelector('#urgencyGroup .selected');
        const aBtn = document.querySelector('#actionGroup .selected');
        
        const payload = {
            action_type: "triage",
            urgency_level: uBtn ? uBtn.dataset.val : "medium",
            recommended_action: aBtn ? aBtn.dataset.val : "clinic_visit",
            reasoning: "Selected via Web UI"
        };

        const res = await fetch('/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        handleStepResult(data);
    });

    const handleStepResult = (data) => {
        btnSubmit.innerText = "Execute Triage";
        resultCard.classList.remove('hidden');
        
        // Populate Score
        const score = data.reward.score;
        document.getElementById('scoreValue').innerText = score.toFixed(2);
        
        // Animate circular progress
        const circle = document.querySelector('.circular-progress');
        // color based on score
        let highlight = 'var(--danger)'; 
        if(score > 0.4) highlight = '#facc15';
        if(score > 0.8) highlight = 'var(--accent)';
        
        const deg = score * 360;
        circle.style.background = `conic-gradient(${highlight} ${deg}deg, rgba(255,255,255,0.1) 0deg)`;

        // Feedback
        document.getElementById('feedbackText').innerText = data.reward.feedback;

        // If done, show truth
        isDone = data.done;
        const truthBox = document.getElementById('truthBox');
        if (isDone) {
            truthBox.classList.remove('hidden');
            document.getElementById('truthList').innerHTML = `
                <li>Urgency: <span style="color:var(--primary);">${data.info.true_urgency}</span></li>
                <li>Action: <span style="color:var(--primary);">${data.info.true_action}</span></li>
            `;
            btnNext.innerText = "Start New Patient";
            btnNext.classList.remove('hidden');
        } else {
            truthBox.classList.add('hidden');
            btnNext.innerText = "Proceed to Next Step";
            btnNext.classList.remove('hidden');
            
            // Background update of state for next step
            populateObservation(data.observation);
            
            // Clear toggles for next step
            document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('selected'));
            checkFormReady();
        }
    };

    btnNext.addEventListener('click', () => {
        btnNext.classList.add('hidden');
        resultCard.classList.add('hidden');
        if (isDone) {
            btnInit.click(); // Load completely new
        } else {
            // Already populated next observation, just wait for form
        }
    });

    setupToggleGroups();
});
