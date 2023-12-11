"""
Created November 2023
@author: Patrick
"""

import pytest
from app import app


# Test done in Web.py mainly simulates that when an app.route is called 
# the renedered template generally has the right set of informations


def test_home_route_redirect(client):
    response = client.get('/')
    assert response.status_code == 302 #Intended Status Code


def test_basic_html_routes(client):
    #/aiExplained.html
    response = client.get('/aiExplained.html')
    assert b'What is AI?' in response.data 
    assert response.status_code == 200  

    #/applicationExplained.html
    response = client.get('/applicationExplained.html')
    assert b'AI application in Daily Lives' in response.data 
    assert response.status_code == 200  

    #/step1.html
    response = client.get('/step1.html')
    assert b'STEP 1' in response.data 
    assert response.status_code == 200  

    #/step2.html
    response = client.get('/step2.html')
    assert b'STEP 2' in response.data
    assert response.status_code == 200  

    #/step3.html
    response = client.get('/step3.html')
    assert b'STEP 3' in response.data
    assert response.status_code == 200  

    #/step4.html
    response = client.get('/step4.html')
    assert b'STEP 4' in response.data
    assert response.status_code == 200  

    #/step5.html
    response = client.get('/step5.html')
    assert b'STEP 5' in response.data
    assert response.status_code == 200  

    #/step6.html
    response = client.get('/step6.html')
    assert b'STEP 6' in response.data
    assert response.status_code == 200  


#Index.html 
def test_home_route(client):
    response = client.get('/index.html')

    assert response.status_code == 200

    assert b'Agro-Ai' in response.data

def test_load_prev_model_route(client):
    response = client.post('/index.html', data={'token-enter': 'token'})

    assert response.status_code == 200

    assert b'LOAD TOKEN' in response.data

def test_save_model(client):
    response = client.get('/save-model')

    #Redirects to Index.html
    assert response.status_code == 302

    with client.session_transaction() as session:
        assert 'model' not in session

# Step4Labeling.html

def test_step4Labeling(client):
    response = client.get('/step4Labeling.html')

    assert response.status_code == 200

    # Check if the rendered template or response data contains the expected text or elements
    assert b'Use the buttons to label an image' in response.data



