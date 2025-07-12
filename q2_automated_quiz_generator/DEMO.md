🎓 Advanced Assessment Generation System - Demo
============================================================

📄 Demo: Document Processing
==================================================
✅ Created sample document: sample_machine_learning.txt
✅ Processed document into 1 chunks
Add of existing embedding ID: 8f2ea751ca934038debbcfe820177862_0
Add of existing embedding ID: 8f2ea751ca934038debbcfe820177862_0
Add of existing embedding ID: 8f2ea751ca934038debbcfe820177862_0
Insert of existing embedding ID: 8f2ea751ca934038debbcfe820177862_0
✅ Added chunks to hybrid RAG system

🔍 Demo: Hybrid RAG Retrieval
==================================================
Number of requested results 6 is greater than number of elements in index 1, updating n_results = 1
✅ Retrieved 1 relevant chunks for query: 'What are the types of machine learning?'

Result 1 (Score: 5.298):
Content: Introduction to Machine Learning Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without be...
Metadata: sample_machine_learning.txt

📝 Demo: Assessment Generation
==================================================
✅ Created assessment request for topic: machine_learning
   Difficulty: DifficultyLevel.MEDIUM
   Question types: ['multiple_choice', 'true_false', 'short_answer']
   Number of questions: 5
Number of requested results 20 is greater than number of elements in index 1, updating n_results = 1


> Entering new AgentExecutor chain...

Invoking: `educational_content_search` with `machine_learning`


{"wikipedia": {"title": "Information about machine_learning", "extract": "Search results for machine_learning would be available in async context.", "source": "wikipedia"}, "khan_academy": {"videos": [{"title": "Video about machine_learning", "description": "Educational content about machine_learning"}], "source": "khan_academy"}}
        {
            "title": "Machine Learning Assessment",
            "description": "This assessment tests your understanding of machine learning concepts, types, and applications.",      
            "questions": [
                {
                    "question_text": "What is the difference between supervised and unsupervised learning?",
                    "question_type": "short_answer",
                    "difficulty": "medium",
                    "learning_objective": "Understand the different types of machine learning",
                    "correct_answer": "Supervised learning is a type of machine learning where the algorithm is trained on labeled data, mapping input features to known output labels. Unsupervised learning, on the other hand, works with unlabeled data and the algorithm tries to find hidden patterns or structures in the data.",
                    "explanation": "Supervised learning uses labeled data to learn, which means the algorithm is given a set of examples (input-output pairs) and it learns to predict the output from the input. In contrast, unsupervised learning does not have labels. It tries to find intrinsic structures in the data like clustering or dimensionality reduction.",
                    "points": 2,
                    "tags": ["machine_learning", "types"]
                },
                {
                    "question_text": "What does 'overfitting' mean in the context of machine learning?",
                    "question_type": "multiple_choice",
                    "difficulty": "medium",
                    "learning_objective": "Identify key concepts in machine learning",
                    "options": ["When a model performs well on training data but poorly on new data", "When a model is too simple to capture the underlying patterns", "When a model performs equally well on training data and new data", "None of the above"],      
                    "correct_answer": "When a model performs well on training data but poorly on new data",
                    "explanation": "Overfitting is a common problem in machine learning where a model is too closely fit to a limited set of data points and may therefore fail to make correct predictions on new data. This is generally due to a model being overly complex, such as having too many parameters relative to the number of observations.",
                    "points": 1,
                    "tags": ["machine_learning", "key_concepts"]
                },
                {
                    "question_text": "True or False: Logistic Regression is used for binary classification tasks.",
                    "question_type": "true_false",
                    "difficulty": "medium",
                    "learning_objective": "Recognize common algorithms and their applications",
                    "is_true": true,
                    "explanation": "True. Logistic Regression is indeed used for binary classification tasks. It is a statistical model that uses a logistic function to model a binary dependent variable.",
                    "points": 1,
                    "tags": ["machine_learning", "algorithms"]
                },
                {
                    "question_text": "Which type of machine learning involves an agent learning to make decisions by taking actions in an environment to achieve maximum cumulative reward?",
                    "question_type": "multiple_choice",
                    "difficulty": "medium",
                    "learning_objective": "Understand the different types of machine learning",
                    "options": ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "None of the above"],    
                    "correct_answer": "Reinforcement Learning",
                    "explanation": "Reinforcement Learning involves an agent that learns to make decisions by interacting with its environment. The agent takes actions, and based on the feedback (rewards or punishments) it gets, it adjusts its behavior with the goal of maximizing the cumulative reward.",
                    "points": 1,
                    "tags": ["machine_learning", "types"]
                },
                {
                    "question_text": "In what field is machine learning NOT commonly used?",
                    "question_type": "multiple_choice",
                    "difficulty": "medium",
                    "learning_objective": "Recognize common algorithms and their applications",
                    "options": ["Healthcare", "Finance", "Marketing", "None of the above"],
                    "correct_answer": "None of the above",
                    "explanation": "Machine learning is commonly used in all the options provided: Healthcare (e.g. disease diagnosis and drug discovery), Finance (e.g. fraud detection and risk assessment), and Marketing (e.g. customer segmentation and recommendation systems).",
                    "points": 1,
                    "tags": ["machine_learning", "applications"]
                }
            ]
        }

> Finished chain.

✅ Generated assessment in 30.28 seconds
   Cache hit: False
   Assessment ID: a72d04e0-7e44-4b22-84ce-7fdb623706b0
   Total points: 6
   Estimated time: 10 minutes

📋 Generated Questions:

Question 1:
   Type: short_answer
   Difficulty: medium
   Text: What is the difference between supervised and unsupervised learning?
   Points: 2
   Explanation: Supervised learning uses labeled data to learn, which means the algorithm is given a set of examples...

Question 2:
   Type: multiple_choice
   Difficulty: medium
   Text: What does 'overfitting' mean in the context of machine learning?
   Options: ['When a model performs well on training data but poorly on new data', 'When a model is too simple to capture the underlying patterns', 'When a model performs equally well on training data and new data', 'None of the above']
   Points: 1
   Explanation: Overfitting is a common problem in machine learning where a model is too closely fit to a limited se...

Question 3:
   Type: true_false
   Difficulty: medium
   Text: True or False: Logistic Regression is used for binary classification tasks.
   Points: 1
   Explanation: True. Logistic Regression is indeed used for binary classification tasks. It is a statistical model ...

Question 4:
   Type: multiple_choice
   Difficulty: medium
   Text: Which type of machine learning involves an agent learning to make decisions by taking actions in an environment to achieve maximum cumulative reward?
   Options: ['Supervised Learning', 'Unsupervised Learning', 'Reinforcement Learning', 'None of the above']
   Points: 1
   Explanation: Reinforcement Learning involves an agent that learns to make decisions by interacting with its envir...

Question 5:
   Type: multiple_choice
   Difficulty: medium
   Text: In what field is machine learning NOT commonly used?
   Options: ['Healthcare', 'Finance', 'Marketing', 'None of the above']
   Points: 1
   Explanation: Machine learning is commonly used in all the options provided: Healthcare (e.g. disease diagnosis an...

🎯 Demo: Difficulty Adjustment
==================================================
✅ Created user performance:
   Score: 3.0/5
   Performance ratio: 0.60

✅ Difficulty adjustment:
   Original difficulty: medium
   Adjusted difficulty: medium

💾 Demo: Caching System
==================================================
✅ Cache Statistics:
   Total requests: 3
   Cache hits: 2
   Cache miss rate: 0.33
   Average response time: 0.003 seconds

✅ Collection Statistics:
   Total documents: 1
   BM25 documents: 3

🎉 Demo completed successfully!