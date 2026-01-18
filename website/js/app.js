/**
 * Aviation MCQ Test Application
 * Main JavaScript file
 */

// ============================================
// App State
// ============================================
const state = {
    subjects: {},
    currentSubject: null,
    questions: [],
    currentQuestionIndex: 0,
    userAnswers: [], // -1 = not answered, 0-3 = selected option
    examFinished: false
};

// Subject icons mapping
const subjectIcons = {
    'COMS': '📡',
    'HPL': '🧠',
    'OPS': '📋',
    'RNAV': '🛰️',
    'default': '📚'
};

// ============================================
// Page Navigation
// ============================================
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    document.getElementById(pageId).classList.add('active');
}

function goHome() {
    showPage('home-page');
}

function showSubjects() {
    showPage('subject-page');
    loadSubjects();
}

function showQuiz() {
    showPage('quiz-page');
}

function showResults() {
    showPage('results-page');
    displayResults();
}

// ============================================
// Data Loading
// ============================================
async function loadSubjects() {
    const grid = document.getElementById('subject-grid');
    grid.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    
    try {
        const response = await fetch('data/subjects.json');
        state.subjects = await response.json();
        
        grid.innerHTML = '';
        
        for (const [code, subject] of Object.entries(state.subjects)) {
            const icon = subjectIcons[code] || subjectIcons.default;
            
            const card = document.createElement('div');
            card.className = 'subject-card';
            card.onclick = () => startExam(code);
            card.innerHTML = `
                <div class="subject-icon">${icon}</div>
                <div class="subject-code">${code}</div>
                <div class="subject-name">${subject.name}</div>
                <div class="subject-count">${subject.questionCount} questions</div>
            `;
            grid.appendChild(card);
        }
        
        // Update home stats
        updateHomeStats();
        
    } catch (error) {
        console.error('Error loading subjects:', error);
        grid.innerHTML = '<p style="color: red;">Error loading subjects. Please refresh the page.</p>';
    }
}

function updateHomeStats() {
    const statsEl = document.getElementById('home-stats');
    const totalSubjects = Object.keys(state.subjects).length;
    const totalQuestions = Object.values(state.subjects).reduce((sum, s) => sum + s.questionCount, 0);
    
    statsEl.innerHTML = `
        <div class="stat">
            <div class="stat-number">${totalSubjects}</div>
            <div class="stat-text">Subjects</div>
        </div>
        <div class="stat">
            <div class="stat-number">${totalQuestions}</div>
            <div class="stat-text">Questions</div>
        </div>
    `;
}

async function loadQuestions(subjectCode) {
    const subject = state.subjects[subjectCode];
    if (!subject) return [];
    
    try {
        const response = await fetch(`data/${subject.file}`);
        const questions = await response.json();
        return questions;
    } catch (error) {
        console.error('Error loading questions:', error);
        return [];
    }
}

// ============================================
// Exam Functions
// ============================================
async function startExam(subjectCode) {
    state.currentSubject = subjectCode;
    state.questions = await loadQuestions(subjectCode);
    
    if (state.questions.length === 0) {
        alert('Error loading questions. Please try again.');
        return;
    }
    
    // Shuffle questions
    state.questions = shuffleArray([...state.questions]);
    
    // Reset state
    state.currentQuestionIndex = 0;
    state.userAnswers = new Array(state.questions.length).fill(-1);
    state.examFinished = false;
    
    // Update UI
    document.getElementById('current-subject').textContent = 
        state.subjects[subjectCode].name;
    
    showQuiz();
    displayQuestion();
}

function displayQuestion() {
    const question = state.questions[state.currentQuestionIndex];
    const total = state.questions.length;
    const current = state.currentQuestionIndex + 1;
    
    // Update header
    document.getElementById('question-counter').textContent = 
        `Question ${current} of ${total}`;
    document.getElementById('progress-fill').style.width = 
        `${(current / total) * 100}%`;
    
    // Update question
    document.getElementById('question-number').textContent = 
        `Question ${current}`;
    document.getElementById('question-text').textContent = question.question;
    
    // Update options
    const container = document.getElementById('options-container');
    container.innerHTML = '';
    
    const markers = ['A', 'B', 'C', 'D'];
    
    question.options.forEach((option, index) => {
        const optionEl = document.createElement('div');
        optionEl.className = 'option';
        
        // Check if this option is selected
        if (state.userAnswers[state.currentQuestionIndex] === index) {
            optionEl.classList.add('selected');
        }
        
        optionEl.onclick = () => selectOption(index);
        optionEl.innerHTML = `
            <span class="option-marker">${markers[index]}</span>
            <span class="option-text">${option}</span>
        `;
        container.appendChild(optionEl);
    });
    
    // Update navigation buttons
    document.getElementById('prev-btn').style.visibility = 
        current === 1 ? 'hidden' : 'visible';
    
    const nextBtn = document.getElementById('next-btn');
    if (current === total) {
        nextBtn.textContent = 'Finish Exam';
        nextBtn.onclick = () => confirmEndExam();
    } else {
        nextBtn.innerHTML = 'Next →';
        nextBtn.onclick = () => nextQuestion();
    }
}

function selectOption(index) {
    state.userAnswers[state.currentQuestionIndex] = index;
    
    // Update UI
    document.querySelectorAll('.option').forEach((opt, i) => {
        opt.classList.toggle('selected', i === index);
    });
}

function nextQuestion() {
    if (state.currentQuestionIndex < state.questions.length - 1) {
        state.currentQuestionIndex++;
        displayQuestion();
    }
}

function prevQuestion() {
    if (state.currentQuestionIndex > 0) {
        state.currentQuestionIndex--;
        displayQuestion();
    }
}

function confirmEndExam() {
    document.getElementById('modal-overlay').classList.add('active');
}

function closeModal() {
    document.getElementById('modal-overlay').classList.remove('active');
}

function endExam() {
    closeModal();
    state.examFinished = true;
    showResults();
}

// ============================================
// Results Functions
// ============================================
function displayResults() {
    let correct = 0;
    let wrong = 0;
    let skipped = 0;
    
    state.questions.forEach((question, index) => {
        const userAnswer = state.userAnswers[index];
        if (userAnswer === -1) {
            skipped++;
        } else if (userAnswer === question.correct) {
            correct++;
        } else {
            wrong++;
        }
    });
    
    const total = state.questions.length;
    const percentage = Math.round((correct / total) * 100);
    
    // Update UI
    document.getElementById('results-subject').textContent = 
        state.subjects[state.currentSubject].name;
    document.getElementById('score-value').textContent = percentage;
    document.getElementById('correct-count').textContent = correct;
    document.getElementById('wrong-count').textContent = wrong;
    document.getElementById('skipped-count').textContent = skipped;
    
    // Update icon based on score
    const icon = document.getElementById('results-icon');
    if (percentage >= 80) {
        icon.textContent = '🎉';
    } else if (percentage >= 60) {
        icon.textContent = '👍';
    } else if (percentage >= 40) {
        icon.textContent = '📚';
    } else {
        icon.textContent = '💪';
    }
    
    // Animate score
    animateScore(percentage);
}

function animateScore(target) {
    const scoreEl = document.getElementById('score-value');
    let current = 0;
    const duration = 1000;
    const step = target / (duration / 16);
    
    const animate = () => {
        current += step;
        if (current >= target) {
            scoreEl.textContent = target;
        } else {
            scoreEl.textContent = Math.round(current);
            requestAnimationFrame(animate);
        }
    };
    
    animate();
}

function retakeExam() {
    startExam(state.currentSubject);
}

function reviewAnswers() {
    showPage('review-page');
    displayReview();
}

function displayReview() {
    const container = document.getElementById('review-list');
    container.innerHTML = '';
    
    const markers = ['A', 'B', 'C', 'D'];
    
    state.questions.forEach((question, index) => {
        const userAnswer = state.userAnswers[index];
        const correctAnswer = question.correct;
        
        let status, statusClass;
        if (userAnswer === -1) {
            status = 'Skipped';
            statusClass = 'skipped';
        } else if (userAnswer === correctAnswer) {
            status = 'Correct';
            statusClass = 'correct';
        } else {
            status = 'Wrong';
            statusClass = 'wrong';
        }
        
        const item = document.createElement('div');
        item.className = `review-item ${statusClass}`;
        
        let optionsHtml = question.options.map((opt, i) => {
            let optClass = '';
            let prefix = markers[i] + '. ';
            
            if (i === correctAnswer) {
                optClass = 'correct-answer';
                prefix = '✓ ' + markers[i] + '. ';
            } else if (i === userAnswer && userAnswer !== correctAnswer) {
                optClass = 'user-wrong';
                prefix = '✗ ' + markers[i] + '. ';
            }
            
            return `<div class="review-option ${optClass}">${prefix}${opt}</div>`;
        }).join('');
        
        item.innerHTML = `
            <div class="review-question-header">
                <span class="review-question-num">Question ${index + 1}</span>
                <span class="review-status ${statusClass}">${status}</span>
            </div>
            <div class="review-question-text">${question.question}</div>
            <div class="review-options">${optionsHtml}</div>
        `;
        
        container.appendChild(item);
    });
}

// ============================================
// Utility Functions
// ============================================
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

// ============================================
// Initialize
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    // Preload subjects for home page stats
    loadSubjects().then(() => {
        // Show home page
        showPage('home-page');
    });
});
