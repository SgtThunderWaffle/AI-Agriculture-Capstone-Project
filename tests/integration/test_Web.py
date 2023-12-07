"""
Created November 2023
@author: Patrick
"""

import pytest
from app import app




def test_home_route(client):
    response = client.get('/')
    assert response.status_code == 302 #Intended Status Code
