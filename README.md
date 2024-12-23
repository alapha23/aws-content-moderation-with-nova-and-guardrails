# AWS Content Moderation with Nova and Guardrails

This project leverages AWS Nova and AWS Guardrails to harness content moderation across diverse images and languages.

## Overview

Content moderation is a critical aspect of maintaining a safe and appropriate online environment. Traditional content moderation approaches often face challenges such as language barriers, context-specific nuances, and false positives. AWS Guardrails provides a robust foundation for content filtering, offering features like denied topic detection and profanity filters to block undesirable topics in generative AI applications. However, while effective, Guardrails may still encounter false positives or miss certain instances due to the complexities of language and context. This project aims to address these limitations by combining the advanced natural language processing capabilities of AWS Nova with the robust content filtering capabilities of AWS Guardrails. By leveraging Nova as an additional layer of verification, the solution can double-check content that may have been incorrectly flagged or missed by Guardrails, reducing false positives and ensuring more accurate content moderation.

### Key Features

- **Multi-Language Support**: AWS Nova's language understanding capabilities enable content moderation across multiple languages, ensuring effective moderation even for non-English content.

- **Context-Aware Moderation**: By leveraging Nova's deep learning models, the solution can understand the context and nuances of language, reducing the risk of false positives and ensuring more accurate content moderation.

- **Multimodal Guardrails Enhancement**: While AWS Guardrails provides a solid foundation for multimodal content filtering, this project enhances its capabilities by utilizing Nova as an additional layer of verification. Nova acts as a safeguard, double-checking content that may have been incorrectly flagged or missed by Guardrails.

- **Comprehensive Moderation**: The solution covers a wide range of content moderation tasks, including profanity filtering, denied topic detection, image auditing, and personal information protection, ensuring a safe and appropriate online environment.

## Getting Started

To get started with this project, follow the instructions in the [Getting Started](./docs/getting-started.md) guide. This guide will walk you through the process of setting up the necessary AWS services, configuring the solution, and deploying it to your environment.

## Contributing

We welcome contributions from the community! If you'd like to contribute to this project, please follow the guidelines outlined in the [Contributing](./docs/contributing.md) document.

## License

This project is licensed under the [License](./LICENSE).
