from pydantic import BaseModel, Field, conlist
from typing import List
from datetime import datetime

class SearchRequest(BaseModel):
    search_text : str = Field(description="Search query")
    model_config = {
        "json_schema_extra": {
            "example": {
                "search_text": "Hello"
            }
        }
    }

class Text(BaseModel):
    content:str

class Document(BaseModel):
    chunk_id: str
    parent_id: str
    content: str
    title: str
    url: str
    filepath: str
    vector: conlist(float, min_length=1536, max_length =1536)  # type: ignore
    doc: datetime

class Source(BaseModel):
    title : str  = Field(description="Document title")
    url : str = Field(description="URL to the relevant document")

class Followup(BaseModel):
    query : str = Field(description="Suggested follow up query")
    button : str = Field(description="above question but in 2 words only for UI purposes")

class Response(BaseModel):
    content : str = Field(description="""Answer to the query, based only on the context provided, in rich text format.
                          Refer to your previous interactions with the user under conversation history for additional context.
                          Structure the answer using paragraphs, bulleted points or a numbered list (with each point starting in a new line) where necessary and highlight key points/words.
                          Leave a line before the start of sub-bulleted points.
                          Use appropriate spacing and multiple paragraph formatting to make the answer look readable.
                          If multiple documents contain relevant information, combine the information coherently.
                          If answer to query is not contained in provided context answer that you can't answer query based on provided context.
                          Use in line citations in [[single digit number assigned to each document that is consistent every time the same doc is cited]](doc_url) format while referencing information from context documents.
                          """)
    source : List[Source] = Field(default_factory=list,
                                  description="""Array of all relevant Source objects.
                                  Only include sources that were used to answer the query and order them based on descending order of relevance.
                                  Leave this field empty only for greetings, while asking clarifying questions or when provided context doesn't contain answer to the query.
                                  """)

class OutputFormat(BaseModel):
    query : str = Field(description="User Query")
    response : List[Response]
    followup : List[Followup] = Field(default_factory=list)