from typing import Dict
from config import config
import os
import streamlit as st
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_aws import ChatBedrock
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Unsupported Windows version")

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "{system}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def getSystemPrompt() -> str:
    system_message = (
        """
            Please generate a comprehensive set of complex JSON payload test cases for the uploaded document. 
            Focus exclusively on creating sophisticated, technically challenging test cases that thoroughly validate the complex functionalities of the platform.

            For each test case, use the following structured format:

            ```
            {
            "Test Case [Number]": [Test case Descriptive name focusing on complex scenario],
            "apiEndpoint": "/api/v1/resource/subresource",
            "httpMethod": "METHOD",
            "testDescription": "Detailed description of the complex scenario being tested",
            "dependencies": ["Any prerequisite API calls or data setup required"],
            "requestHeaders": {
                "Authorization": "Bearer [token]",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Additional-Headers": "As needed for complex scenarios"
            },
            "requestParameters": {
                "queryParam1": "value1",
                "queryParam2": "value2"
            },
            "requestBody": {
                // Detailed JSON object with nested structures, arrays, and various data types
            },
            "expectedResponseCode": 200,
            "expectedResponseHeaders": {
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
                "Other-Headers": "As appropriate"
            },
            "expectedResponseBody": {
                // Detailed expected response JSON
            },
            "validationChecks": [
                "Detailed list of checks to perform on the response",
                "Schema validation requirements",
                "Business logic validation points",
                "Data consistency checks",
                "Performance thresholds"
            ],
            "edgeCaseConsiderations": [
                "Boundary conditions",
                "Race conditions",
                "Timeout scenarios",
                "Load considerations"
            ],
            "testData": {
                "description": "Description of test data requirements",
                "setup": "Data setup needs",
                "cleanup": "Data cleanup needs"
            }
            }
            ```

            Please create test cases with the following characteristics:

            1. **Highly Complex Data Structures**:
            - Deeply nested JSON objects (at least 4-5 levels deep)
            - Arrays of complex objects with varying structures
            - Mixed data types (strings, numbers, booleans, dates, enums, custom objects)
            - Conditional fields that appear only under specific circumstances

            2. **Sophisticated Business Logic**:
            - Complex calculation validations
            - Multi-condition business rules
            - Interdependent field validations
            - Time-series data with temporal validations

            3. **Advanced Testing Scenarios**:
            - Multi-step API workflows with state dependencies
            - Batch operations with partial success/failure scenarios
            - Concurrency testing with potential race conditions
            - Performance edge cases with large payloads

            4. **Comprehensive Error Testing**:
            - Detailed validation error responses
            - Business rule violation scenarios
            - Security constraint testing
            - Graceful degradation under stress

            Focus on the complex aspects of each section in the requirements document, particularly:
            - Advanced analytics APIs with complex statistical models
            - Brand positioning algorithms with multidimensional data
            - Feature optimization with multiple constraints
            - Real-time data streaming with complex transformation rules
            - Multi-tenant security scenarios with various permission levels
            - Large dataset handling with pagination and filtering

            Generate highly complex test cases for each of the sections (major functional areas) and subsections and include detailed 
            request/response payloads that reflect the complexity of enterprise-grade market research platform APIs.

            Please organize the test cases by the document's section numbers to ensure comprehensive coverage of all 
            API functionality described in the requirements document.

            CRITICAL INSTRUCTION: Generate test cases in a strictly sequential manner, covering each section of the document in order from the first section to the last. 
            - Begin with Section [i] and complete all test cases for that section
            - Proceed to Section [i+1] only after fully completing Section [i]

            For requests other than JSON payload test cases requests (e.g., generating test data in tabular format):\n"
            "   - Provide the requested information without JSON data.\n"
            "   - Provide structured test data in markdown table format.\n"
            "   - Use appropriate formatting (e.g., markdown tables for tabular data).\n"
            "   - Ensure realistic test data that aligns with business logic.\n"
            "   - Cover different data types, including numerical, text, date/time, and categorical values.\n\n"

            "   Example Format:\n"
            "   | Column1 | Column2 | Column3 |\n"
            "   |---------|---------|--------|\n"
            "   | Data1   | Data2   | Data3   |\n\n"

            "   Example:\n"
            "   | Order ID | Customer Name | Order Date  | Total Amount |\n"
            "   |---------|--------------|------------|-------------|\n"
            "   | 1001    | John Doe      | 2024-01-15 | 250.00      |\n"
            "   | 1002    | Alice Smith   | 2024-02-10 | 500.50      |\n\n"

            "3. For general questions or requests:\n"
            "   - Provide clear, concise answers without unnecessary formatting.\n\n"
            
            "Always ensure your responses are **well-structured, accurate, and tailored** to the user's specific request. "
        """
    )
    
    return system_message


def getSystemPromptGherkinFormat() -> str:
    system_message = (
        """
        Please generate a comprehensive set of complex Gherkin test scenarios for the uploaded document. 
        Focus exclusively on creating sophisticated, technically challenging test cases that thoroughly validate the complex functionalities of the platform.

        For each test case, use the following Gherkin format:

        ```gherkin
        Feature: [Feature Name Based on Document Section]

        Background:
            Given the user has a valid API key
            And the user is authenticated with the Market Survey Platform API

        Scenario: [Test case Descriptive name focusing on complex scenario]
            Given [preconditions and setup]
            When [actions performed against the API endpoint]
            Then [expected outcomes]
            And [additional validation checks]

        Scenario Outline: [Parameterized test case name for data-driven testing]
            Given [preconditions with variables]
            When [actions with variables]
            Then [expected outcomes with variables]

            Examples:
            | variable1 | variable2 | expectedResult |
            | value1    | value2    | result1        |
            | value3    | value4    | result2        |
        Please create test scenarios with the following characteristics:

        Highly Complex Data Structures:


        Deeply nested data objects (at least 4-5 levels deep)
        Arrays of complex objects with varying structures
        Mixed data types (strings, numbers, booleans, dates, enums, custom objects)
        Conditional fields that appear only under specific circumstances


        Sophisticated Business Logic:


        Complex calculation validations
        Multi-condition business rules
        Interdependent field validations
        Time-series data with temporal validations


        Advanced Testing Scenarios:


        Multi-step API workflows with state dependencies
        Batch operations with partial success/failure scenarios
        Concurrency testing with potential race conditions
        Performance edge cases with large payloads


        Comprehensive Error Testing:


        Detailed validation error responses
        Business rule violation scenarios
        Security constraint testing
        Graceful degradation under stress

        Focus on the complex aspects of each section in the requirements document, particularly:

        Advanced analytics APIs with complex statistical models
        Brand positioning algorithms with multidimensional data
        Feature optimization with multiple constraints
        Real-time data streaming with complex transformation rules
        Multi-tenant security scenarios with various permission levels
        Large dataset handling with pagination and filtering

        Generate highly complex test scenarios for each of the sections (major functional areas) and subsections and include detailed descriptions that reflect the complexity of enterprise-grade market research platform APIs.
        Please organize the test scenarios by the document's section numbers to ensure comprehensive coverage of all API functionality described in the requirements document.
        CRITICAL INSTRUCTION: Generate test scenarios in a strictly sequential manner, covering each section of the document in order from the first section to the last.

        Begin with Section [i] and complete all test scenarios for that section
        Proceed to Section [i+1] only after fully completing Section [i]
        """
    )

    return system_message


def getSystemPromptPlaywright() -> str:
    system_message = (
        """
            Based on the JSON payloads from my API test cases, please:

            I need to create Playwright test scripts for API testing based on the uploaded document. Generate structured 
            Playwright code for API test cases that cover the following areas from the document:

            Create a complete Playwright test suite that:
            - Sets up the test environment with proper authentication
            - Implements test cases that validate the requirements and acceptance criteria
            - Includes proper assertions to validate the acceptance criteria
            - Handles error conditions and edge cases
            - Follows best practices for API testing with Playwright

            The test script should include:
            - Proper imports and configuration setup
            - Authentication/authorization handling
            - Test fixtures where appropriate
            - Appropriate test data generation or loading
            - Clear test case organization with descriptive test names
            - Comprehensive assertions that align with the acceptance criteria
            - Error handling and reporting
            - Any necessary setup/teardown operations

            Include comments explaining the purpose of each test and how it relates to the specific requirements in the document.

            The code should be production-ready, following best practices for Playwright API testing, with proper error handling, 
            timeouts, and retries where appropriate.

            Please focus on creating a complete, working test suite for one specific module rather than partial coverage of 
            multiple modules. I'd like the code to be well-structured and maintainable.
        """
    )

    return system_message


def getSystemPromptCodeGenerator() -> str:
    system_message = (
        """
            I need to create C# code for API testing based on the uploaded document. Please generate professional-quality 
            C# code that implements test cases for these APIs with the following specifications:

            Create a complete C# test suite that:
            - Uses appropriate C# testing frameworks (NUnit, xUnit, or MSTest)
            - Implements HttpClient or RestSharp for API interactions
            - Properly handles authentication and authorization
            - Validates the specific requirements and acceptance criteria for the chosen module
            - Includes robust assertions and validation
            - Handles error conditions and edge cases properly

            The C# code should include:
            - Proper namespaces, classes, and method organization
            - Authentication/authorization implementation
            - Test fixture setup and teardown
            - Appropriate test data handling
            - Clear test method naming and organization
            - Comprehensive assertions aligned with acceptance criteria
            - Exception handling and proper logging
            - Appropriate setup/teardown operations

            Include these specific implementation details:
            - Base URL for the API: [Include your API base URL]
            - Authentication method: [Specify OAuth, API key, JWT, etc.]
            - Any required headers or tokens
            - Test environment configuration
            - Any specific C# libraries or frameworks you want to use

            Add XML documentation comments explaining the purpose of each test and how it maps to the requirements in the document.

            Include appropriate error handling and logging in the implementation.

            Please organize the code following clean architecture principles with:
            - Models in a separate namespace
            - Services in their own namespace
            - Extensions and helpers in utility namespaces
            - Test projects properly structured

            The code should be production-ready, following C# coding standards and best practices for API testing, 
            with proper error handling, timeouts, and retry logic where appropriate.

            Please focus on creating a complete, working test suite for one specific module rather than partial 
            coverage of multiple modules. I need code that is well-structured, maintainable, and follows C# best 
            practices.

            Generate highly complex C# code considering each of the sections (major functional areas) and subsections.
        """
    )

    return system_message


def getSystemPromptUnitTesting() -> str:
    system_message = (
        """
            Based on the generated C# code, please generate comprehensive C# unit test cases for API testing. I need 
            a complete set of unit tests that ranges from simple to complex scenarios with high code coverage. 
            
            The tests should:

            Create a structured test suite using a C# unit testing framework (xUnit, NUnit, or MSTest) that includes:
            - Simple tests: Basic validation of endpoints, status codes, and response formats
            - Intermediate tests: Data validation, business logic verification, and edge cases
            - Complex tests: Integration scenarios, workflow sequences, and exceptional conditions
            - Performance-oriented tests: Response time validation and concurrent request handling

            Implement proper test organization with:
            - Test classes organized by functional area
            - Clear naming convention (e.g., "When_[Condition]_Should_[ExpectedResult]")
            - Appropriate test categorization/grouping
            - Setup and teardown methods for test context
            - Shared test fixtures and data preparation

            Include thorough test coverage with:
            - Happy path scenarios
            - Boundary value testing
            - Negative testing with invalid inputs
            - Error condition handling
            - Security-related scenarios (permissions, authentication)
            - Data persistence verification
            - State transition validation

            Implement best practices:
            - Mocking of external dependencies
            - Parameterized tests for data-driven scenarios
            - Clear assertions with descriptive failure messages
            - Isolation between tests
            - Appropriate use of setup/teardown
            - Thorough comments explaining test purpose and requirements coverage

            Show examples of handling different response types (JSON, XML) and validating them against schemas or 
            expected structures.

            Include proper exception handling and testing for API error responses.

            Please ensure the tests are maintainable, follow C# coding standards, and provide high coverage of the 
            requirements and acceptance criteria specified in the document. Each test should be clearly linked to 
            specific requirements from the document via comments.

            For the unit tests, use xUnit (or NUnit/MSTest if preferred) with Moq for mocking dependencies.
        """
    )

    return system_message


class ChatModel:
    def __init__(self, model_name: str, model_kwargs: Dict, rag_prompt: str):
        self.model_config = config["models"][model_name]
        self.model_id = self.model_config["model_id"]
        self.rag_prompt = rag_prompt

        self.model_kwargs = model_kwargs

        self.llm = PROMPT | ChatBedrock(model_id=self.model_id, 
                                        model_kwargs=self.model_kwargs, 
                                        streaming=True,
                                        # aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                                        # aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                                        region_name=os.getenv("AWS_DEFAULT_REGION")
                                        )


    def format_prompt(self, tab: int) -> str:
        print("Using prompt of tab 1")
        system_message = (
            f"Uploaded Document: {self.rag_prompt}\n\n"
            f"{getSystemPrompt()}"
        )

        if tab == 2:
            print("Using prompt of tab 1")
            system_message = (
                f"Uploaded Document: {self.rag_prompt}\n\n"
                f"{getSystemPromptGherkinFormat()}"
            )
        elif tab == 3:
            print("Using prompt of tab 3")
            system_message = (
                f"Uploaded Document: {self.rag_prompt}\n\n"
                f"{getSystemPromptPlaywright()}"
            )
        elif tab == 4:
            print("Using prompt of tab 4")
            system_message = (
                f"Uploaded Document: {self.rag_prompt}\n\n"
                f"{getSystemPromptCodeGenerator()}"
            )
        elif tab == 5:
            print("Using prompt of tab 5")
            system_message = (
                f"Uploaded Document: {self.rag_prompt}\n\n"
                f"{getSystemPromptUnitTesting()}"
            )

        return system_message
        

    


