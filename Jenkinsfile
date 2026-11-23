pipeline {
    agent any

    stages {
        stage('Setup Python & Node Environments') {
            steps {
                sh '''
                    python3 -m venv .venv || true
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install pytest pandas numpy requests python-dotenv
                    cd dashboard && npm ci || npm install
                '''
            }
        }

        stage('Run Automated Pipeline Tests (Pytest)') {
            steps {
                sh '''
                    . .venv/bin/activate
                    pytest -v tests/test_pipeline.py
                '''
            }
        }

        stage('Run 20-Seed Statistical Verification') {
            steps {
                sh '''
                    . .venv/bin/activate
                    python steps/step15_robust_20seed_eval.py
                '''
            }
        }

        stage('Deploy Node.js Operator Dashboard') {
            steps {
                sh '''
                    cd dashboard
                    pkill -f "node server.js" || true
                    nohup node server.js > ../outputs/dashboard.log 2>&1 &
                '''
            }
        }
    }
}