"""
Unit tests for Flask routes
"""

import unittest
import json
from app import create_app

class TestRoutes(unittest.TestCase):
    
    def setUp(self):
        """Set up test client"""
        self.app = create_app({'TESTING': True})
        self.client = self.app.test_client()
        
        # Sample valid input
        self.valid_input = {
            'square_feet': 2500,
            'bedrooms': 4,
            'bathrooms': 3,
            'age': 5,
            'lot_size': 8000,
            'garage_spaces': 2,
            'neighborhood_quality': 8,
            'distance_to_city_center': 3.5
        }
    
    def test_index_route(self):
        """Test main page loads"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'House Price Predictor', response.data)
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_api_info(self):
        """Test API info endpoint"""
        response = self.client.get('/api/info')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('service', data)
        self.assertIn('endpoints', data)
    
    def test_predict_valid_input(self):
        """Test prediction with valid input"""
        response = self.client.post(
            '/predict',
            data=json.dumps(self.valid_input),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('prediction', data)
        self.assertIn('confidence_score', data)
        self.assertGreater(data['prediction'], 0)
    
    def test_predict_missing_field(self):
        """Test prediction with missing field"""
        invalid_input = self.valid_input.copy()
        del invalid_input['square_feet']
        
        response = self.client.post(
            '/predict',
            data=json.dumps(invalid_input),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')
        self.assertIn('error', data)
    
    def test_predict_invalid_range(self):
        """Test prediction with out-of-range values"""
        invalid_input = self.valid_input.copy()
        invalid_input['bedrooms'] = 10  # Out of valid range
        
        response = self.client.post(
            '/predict',
            data=json.dumps(invalid_input),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')
    
    def test_predict_no_data(self):
        """Test prediction with no data"""
        response = self.client.post(
            '/predict',
            data=json.dumps({}),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
    
    def test_predict_invalid_type(self):
        """Test prediction with invalid data type"""
        invalid_input = self.valid_input.copy()
        invalid_input['square_feet'] = 'not_a_number'
        
        response = self.client.post(
            '/predict',
            data=json.dumps(invalid_input),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()