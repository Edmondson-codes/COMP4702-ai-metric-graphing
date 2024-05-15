from crewai import Agent, Crew, Task
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langchain_anthropic import ChatAnthropic
import os

Claude_3 = os.environ["ANTHROPIC_API_KEY"] = "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"

LLM = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")

search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Find information online. Useful for finding the answers to things your not sure about or how to do."
    )
]


# Define agents
# requirements_analyst = Agent(
#     role='Requirements Analyst',
#     goal='Gather and analyze user requirements for software development',
#     backstory='Experienced in translating user needs into detailed software requirements.',
#     verbose=True,
#     allow_deligation=False,
#     llm=LLM,
#     tools=tools
# )

# software_designer = Agent(
#     role='Software Designer',
#     goal='Design software architecture based on requirements',
#     backstory='Skilled in designing scalable and robust software architectures.',
#     verbose=True,
#     allow_deligation=False,
#     llm=LLM,
#     tools=tools
# )

software_developer = Agent(
    role='Software Developer',
    goal='Develop and implement software based on the design',
    backstory='Proficient in multiple programming languages and development frameworks.',
    verbose=True,
    allow_deligation=False,
    llm=LLM,
    tools=tools
)

# Create tasks
# gather_requirements_task = Task(
#     description='''break down the requirements of an AI application with a flutter front-end and node.js server-side.
#     It should allow users to select Claude or GPT4 as the ai, select from a store of customized models
#     (where users have uploaded model system prompts and knowledge bases) and create customized models for the store.
#     It should also integrate stripe and have a payment page so that the user can put fund into their account to use with ai models.
#      the user should be charged per token generated and have that amount deducted from their accouny.
#     If the user runs out of funds, it should not allow them to use the ai until they have topped up their account with funds again.''',
#     expected_output='A document containing all software requirements.',
#     agent=requirements_analyst
# )

# design_software_task = Task(
#     description='Design the software architecture based on the gathered requirements.',
#     expected_output='Software design document including architecture and technologies used.',
#     agent=software_designer
# )

develop_software_task = Task(
    description="""finish of this code, ensuring it can handle 10 inputs. The inputs are from: ""Species", "Population", "Latitude", "Longitude", 'Year_start', "Year_end", "Temperature", "Replicate", "Thorax_length", "l2", "l3p", "l3d", "lpd", "l3", 'w1', 'w2', 'w3', "wing_loading"". The code is:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# https://github.com/nathasha-naranpanawa/COMP4702_2024/blob/main/PracW3.ipynb

# D._aldrichi = 0     D._buzzatii = 1

# @title Classification
# load the dataset
data = pd.read_csv("../83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv", names=["Species", "Population", "Latitude", "Longitude", 'Year_start', "Year_end", "Temperature", "Replicate", "Thorax_length", "l2", "l3p", "l3d", "lpd", "l3", 'w1', 'w2', 'w3', "wing_loading"])


#split dataset into features and targets
X = data.drop(["Species", "Population", "Latitude", "Longitude", 'Year_start', "Year_end", "Temperature", "Replicate"], axis=1)
y = data["Species"]

# split into train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# @title Build and train a classifier

# create classifier
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# fit the model on the training data
knn.fit(X_train, y_train)

# make predictions on training data and calculate training Misclassification Rate
y_train_pred = knn.predict(X_train)
train_misclassification_rate = 1 - accuracy_score(y_train, y_train_pred)
print(f"Training Misclassification Rate: {train_misclassification_rate:.2f}")

# make predictions on the test data and calculate test Misclassification Rate
y_test_pred = knn.predict(X_test)
test_misclassification_rate = 1 - accuracy_score(y_test, y_test_pred)
print(f"Test Misclassification Rate: {test_misclassification_rate:.2f}")

# @title Plotting the decision boundary for training data

# Perform PCA to reduce dimensionality to 2 components
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Generate a meshgrid of points to cover the feature space
h = 0.02
x_min, x_max = X_train_pca[:, 0].min() - 0.5, X_train_pca[:, 0].max() + 0.5
y_min, y_max = X_train_pca[:, 1].min() - 0.5, X_train_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict class labels for the points in the meshgrid
Z = knn.predict(pca.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Plot the decision regions
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

# Plot the training data points
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
plt.scatter(X_train_pca""",
    expected_output='Completed software ready for testing.',
    agent=software_developer
)



# TODO: make software tester after running developer

# # Instantiate the crew with a sequential process
# software_development_crew = Crew(
#     agents=[requirements_analyst, software_designer, software_developer],
#     tasks=[gather_requirements_task, design_software_task, develop_software_task],
#     verbose=True
# )

# Instantiate the crew with a sequential process
software_development_crew = Crew(
    agents=[software_developer],
    tasks=[develop_software_task],
    verbose=True
)

# Execute the process
result = software_development_crew.kickoff()
print(result)


"""

1. API Layer:
   - This layer will expose RESTful APIs for the front-end and other clients to interact with the application.
   - It will handle HTTP requests, perform input validation, and route requests to the appropriate business logic layer.
   - The API layer will be responsible for serializing and deserializing data, as well as handling authentication and authorization.

2. Business Logic Layer:
   - This layer will contain the core application logic, rules, calculations, and operations that define the application's functionality.
   - It will be responsible for processing data, enforcing business rules, and coordinating the flow of data between the API layer and the data access layer.
   - The business logic layer will be designed using appropriate design patterns, such as the Repository Pattern or the Service Layer Pattern, to promote code reusability, testability, and maintainability.

3. Data Access Layer:
   - This layer will handle all interactions with the database, including reading and writing data.
   - It will abstract the underlying database implementation details from the business logic layer, promoting a loosely coupled architecture.
   - The data access layer will translate between the application's data structures and the database schema, ensuring data consistency and integrity.
   - It may utilize an Object-Relational Mapping (ORM) tool or a micro-ORM library to simplify database operations.

4. Integration Layer:
   - This layer will be responsible for integrating with external services and AI models.
   - For AI model integration, it will handle tasks such as preprocessing data, sending requests to the AI model, and processing the model's output.
   - For external service integration (e.g., Stripe payment gateway), it will encapsulate the logic for making API calls, handling responses, and managing any required authentication or authorization.
   - The integration layer will promote loose coupling between the application's core logic and external dependencies, making it easier to swap out or update integrations as needed.

Additionally, the back-end architecture will incorporate cross-cutting concerns such as logging, caching, error handling, and monitoring. Appropriate design patterns and principles, such as Dependency Injection, Separation of Concerns, and the Single Responsibility Principle, will be followed to ensure a modular, maintainable, and scalable architecture.
"""


"""
1. Introduction
   - This document outlines the software architecture for the AI application with a Flutter front-end and Node.js back-end, as per the provided Software Requirements Document.

2. Architecture Overview
   - The application follows a client-server architecture with a separation of concerns between the front-end and back-end components.
   - The front-end is developed using Flutter, providing a responsive and user-friendly interface for interacting with the AI models and managing user accounts.
   - The back-end is built with Node.js and Express.js, handling the application logic, data processing, and integration with external services.

3. Front-end Architecture (Flutter)
   - User Interface Layer
     - Developed using Flutter's widget-based UI framework
     - Responsible for rendering the user interface and handling user interactions
     - Utilizes state management solutions like Provider or Bloc for efficient state management
   - Service Layer
     - Encapsulates the communication with the back-end APIs
     - Handles API requests and responses
     - Integrates with the AI models for features like natural language processing, recommendations, etc.
   - State Management Layer
     - Manages the application state and data flow
     - Implements reactive programming principles for efficient state updates and UI rendering
     - Utilizes Provider, Bloc, or similar state management solutions

4. Back-end Architecture (Node.js)
   - API Layer
     - Exposes RESTful APIs for the front-end to consume
     - Handles HTTP requests and responses
     - Implements input validation, rate limiting, and security measures
   - Business Logic Layer
     - Encapsulates the core application logic and business rules
     - Manages user authentication, authorization, and account management
     - Integrates with the AI models for data processing and decision-making
   - Data Access Layer
     - Interacts with the database (e.g., MongoDB) for data storage and retrieval
     - Implements data access patterns like Repository or Data Mapper
     - Handles data transformations and mappings between the application and database models
   - Integration Layer
     - Integrates with external services like the Stripe payment gateway
     - Handles secure communication and data exchange with third-party APIs
     - Implements error handling, retries, and fallback mechanisms for reliable integrations

5. AI Model Integration
   - The application integrates with the Claude and GPT-4 AI models provided by Anthropic.
   - The AI models are consumed through APIs or SDKs provided by Anthropic.
   - The front-end and back-end components interact with the AI models as needed for tasks like natural language processing, recommendations, and data analysis.
   - Customized models created by users are stored and managed in a dedicated model store.

6. Payment Gateway Integration
   - The application integrates with the Stripe payment gateway for processing user payments.
   - The back-end handles the communication with the Stripe API for initiating and completing transactions.
   - User payment information is securely stored and handled in compliance with industry standards.
   - Users are charged per token generated by the AI models, and the amount is deducted from their account balance.

7. Database
   - A database (e.g., MongoDB) is used for storing and managing application data, such as user accounts, customized models, and payment information.
   - The back-end interacts with the database through the Data Access Layer.
   - Appropriate indexing, sharding, and caching strategies are implemented for optimal performance and scalability.

8. Authentication and Authorization
   - User authentication and authorization mechanisms are implemented using industry-standard practices like JWT or OAuth.
   - The back-end handles user registration, login, and account management.
   - Role-based access control (RBAC) is implemented to manage user permissions and access to application features.

9. Logging and Monitoring
   - Logging and monitoring systems are integrated into both the front-end and back-end components.
   - Application logs and metrics are collected and analyzed for performance monitoring, error tracking, and usage insights.
   - Tools like Elasticsearch, Logstash, and Kibana (ELK stack) or cloud-based logging services can be utilized.

10. Deployment and DevOps
    - The application is deployed to cloud hosting platforms like AWS, Heroku, or DigitalO
"""