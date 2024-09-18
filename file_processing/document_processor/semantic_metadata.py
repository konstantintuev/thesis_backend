# Thank you o1-preview

from typing import Optional, List, Dict

from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field

from file_processing.llm_chat_support import LLMTemp, get_llm, small_llm_json_response


# Thank you o1-preview

# Too complicated and exact for small models like Llama 8B
class ManualData(BaseModel):
    # **Identifiers** (Optional)
    product_name: str = Field(..., description="Full name of the product")
    serial_number: Optional[str] = Field(None, description="Unique serial number of the product")
    model_number: Optional[str] = Field(None, description="Model number of the product")

    # **Version Information** (Optional)
    version: Optional[str] = Field(None, description="Version or revision of the manual")

    # **Dimensions** (Optional)
    length: Optional[str] = Field(None, description="Length of the product, e.g., '100cm'")
    width: Optional[str] = Field(None, description="Width of the product")
    height: Optional[str] = Field(None, description="Height of the product")
    weight: Optional[str] = Field(None, description="Weight of the product")

    # **Material Properties** (Optional)
    material_properties: Optional[Dict[str, str]] = Field(
        None, description="Properties of materials used, e.g., {'Aluminum': 'Density: 2.7g/cm³'}"
    )

    # **Performance Parameters** (Optional)
    max_load: Optional[str] = Field(None, description="Maximum load capacity, e.g., '500kg'")
    efficiency: Optional[str] = Field(None, description="Efficiency rating, e.g., '85%'")
    power_consumption: Optional[str] = Field(None, description="Power consumption details")

    # **Components** (Optional)
    components: Optional[List[str]] = Field(
        None, description="List of main components, e.g., ['Motor', 'Sensor']"
    )

    # **Operating Conditions** (Optional)
    temperature_range: Optional[str] = Field(
        None, description="Operational temperature range, e.g., '-20°C to 50°C'"
    )
    humidity_range: Optional[str] = Field(None, description="Operational humidity range")

    # **Compliance Information** (Optional)
    safety_standards: Optional[List[str]] = Field(
        None, description="Safety standards complied with, e.g., ['ISO 13849-1']"
    )
    certifications: Optional[List[str]] = Field(
        None, description="Certifications obtained, e.g., ['CE', 'UL']"
    )

    # **Software Information** (Optional)
    firmware_version: Optional[str] = Field(None, description="Firmware version installed")
    software_update_date: Optional[str] = Field(None, description="Date of last software update")

    # **Networking Information** (Optional)
    protocols: Optional[List[str]] = Field(
        None, description="Supported communication protocols, e.g., ['TCP/IP', 'HTTP']"
    )
    ports: Optional[List[int]] = Field(None, description="Network ports used")

    # **Error Codes** (Optional)
    error_codes: Optional[Dict[str, str]] = Field(
        None, description="Error codes and their descriptions, e.g., {'E001': 'Overheating detected'}"
    )

    # **Maintenance Schedule** (Optional)
    maintenance_tasks: Optional[List[str]] = Field(
        None, description="Scheduled maintenance tasks, e.g., ['Monthly: Inspect belts']"
    )

    # **User Interface Details** (Optional)
    control_panel_description: Optional[str] = Field(
        None, description="Description of control panels and interfaces"
    )

    # **Security Features** (Optional)
    user_roles: Optional[List[str]] = Field(
        None, description="User roles defined, e.g., ['Admin', 'User']"
    )
    permissions: Optional[Dict[str, List[str]]] = Field(
        None, description="Permissions associated with each role"
    )

    # **Localization Details** (Optional)
    supported_languages: Optional[List[str]] = Field(
        None, description="Supported languages, e.g., ['en-US', 'es-ES']"
    )
    default_language: Optional[str] = Field(None, description="Default language setting")

    # **Life Cycle Information** (Optional)
    expected_lifespan: Optional[str] = Field(
        None, description="Expected operational lifespan, e.g., '10 years'"
    )
    warranty_period: Optional[str] = Field(None, description="Warranty period offered")

    # **Manufacturing Data** (Optional)
    process_specifications: Optional[str] = Field(
        None, description="Key manufacturing process specifications"
    )

    # **Additional Notes** (Optional)
    additional_notes: Optional[str] = Field(
        None, description="Any other relevant information from the manual"
    )


# Too complicated and exact for small models like Llama 8B
def extract_semantic_metadata(technical_document: str):
    try:
        system_message_template = SystemMessagePromptTemplate.from_template(
            "You a technical writer and need to create JSON metadata for a technical document.\n"
            "Return a JSON object describing the most important characteristics of the product as metadata.\n"
            "When preparing the metadata json take inspiration from Industry 4.0 and metadata for a digital twin of a product!\n"
            "Focus on product specifications and descriptions of processes. You can ignore irrelevant info!\n"
            "Only answer in JSON."
        )

        human_message_template = HumanMessagePromptTemplate.from_template(
            "Here is the technical document:\n\n{document}"
        )

        chat_prompt = ChatPromptTemplate.from_messages([system_message_template, human_message_template])

        structured_llm = get_llm(LLMTemp.CONCRETE).with_structured_output(ManualData)

        rewrite_query = (chat_prompt | structured_llm)

        res = rewrite_query.invoke({"document": technical_document})
        return res.to_dict()
    except BaseException as e:
        print(e)
        return {}


def extract_semantic_metadata_together(technical_document: str):
    try:
        system_message_template = SystemMessagePromptTemplate.from_template(
            """You are a technical writer tasked with creating JSON metadata for a technical document. Your goal is to return a JSON object describing the most important characteristics of the component as metadata. When preparing the metadata, take inspiration from Industry 4.0 and digital twin concepts. Focus on product specifications, component details, operating conditions, performance parameters, and maintenance data relevant to various components like brakes, rotors, motors, displays, controllers, etc. Ignore irrelevant information.

Please follow this general schema for the JSON response:

{{
    "product_info": {{
        "product_name": "string",
        "serial_number": "string",
        "model_number": "string",
        "version": "string"
    }},
    "component_type": "string",  // e.g., "Motor", "Brake", "Display"
    "technical_specifications": {{
        // Include key specifications relevant to the component type
    }},
    "dimensions": {{
        "length": "string",
        "width": "string",
        "height": "string",
        "weight": "string"
    }},
    "material_properties": {{
        "material_name": "string",
        "material_type": "string",
        "density": "string"
    }},
    "performance_parameters": {{
        // Include performance metrics relevant to the component type
    }},
    "compatibility": {{
        "other_component_names": ["string"],
        "voltage": "string",
        "current": "string",
        "communication_protocols": ["string"]
    }},
    "components_included": [
        {{
            "name": "string",
            "quantity": "integer",
            "specs": {{
                "key": "value"
            }}
        }}
    ],
    "operating_conditions": {{
        "temperature_range": "string",
        "humidity_range": "string",
        "ingress_protection": "string"
    }},
    "installation_instructions": "string",
    "maintenance_schedule": [
        {{
            "interval": "string",
            "tasks": [
                "string"
            ]
        }}
    ],
    "software_info": {{
        "firmware_version": "string",
        "compatible_apps": ["string"],
        "update_date": "string"
    }},
    "certifications": ["string"],
    "warranty": {{
        "period": "string",
        "terms": "string"
    }}
}}

Respond only in JSON format and ensure the response follows the provided structure.
"""
        )

        human_message_template = HumanMessagePromptTemplate.from_template(
            "Here is the technical document:\n\n{document}"
        )

        chat_prompt = ChatPromptTemplate.from_messages([system_message_template, human_message_template])

        msges = chat_prompt.invoke({"document": technical_document}).to_messages()

        msges = [{"role": "user" if msg.type == "human" else "system", "content": msg.content} for msg in msges]

        output = small_llm_json_response(msges)

        return output
    except BaseException as e:
        print(e)
        return {}
