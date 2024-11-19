# **OrdinanceGPT: Architectural Design Document**

## **1. Introduction**
### **1.1 Purpose**
This document describes the architectural design of the ordinance web scraping and extraction tool, focusing on its components, key classes, and their roles within the system.

### **1.2 Audience**
- **Primary:** Model developers working on expanding the capabilities of ordinance extraction.
- **Secondary:** Model developers extending this functionality to other contexts.

### **1.3 Scope**
Covers the backend API design, including key classes, their responsibilities, and interactions.

---

## **2. High-Level Architecture**
### **2.1 System Context**
Points of interaction for OrdinanceGPT:
- **End Users:** Users submit model executions via command line using a configuration file. Users can select specific jurisdictions to focus on.
- **Internet via Web Browser:** The model searches the web for relevant legal documents. The most common search technique is Google Search.
- **LLMs:** The model relies on LLMs (typically ChatGPT) to analyze web scraping results and subsequently extract information from documents.
- **Filesystem:** Stores output files in organized sub-directories and compiles ordinance information into a CSV.

**Diagram:**
```mermaid
architecture-beta
    group model[OrdinanceGPT]

    service input[User Input]
    service scraper(server)[Web Scraper] in model
    service llm(cloud)[LLM Service]
    service web(internet)[Web]
    service ds(disk)[Document Storage]
    service parser(server)[Document Parser] in model
    service out(disk)[Ordinance CSV]

    input:R --> L:scraper
    scraper:T --> L:llm
    scraper:T --> R:web
    scraper:B --> T:ds
    scraper:R --> L:parser
    parser:T --> B:llm
    parser:B --> T:out
```

---
