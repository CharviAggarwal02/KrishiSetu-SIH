import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import axios from 'axios';
import './App.css';

// Import shadcn/ui components
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Textarea } from './components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Alert, AlertDescription } from './components/ui/alert';
import { Badge } from './components/ui/badge';
import { Progress } from './components/ui/progress';
import { Toaster } from './components/ui/toaster';
import { useToast } from './hooks/use-toast';

// Import Lucide React icons
import { 
  Sprout, 
  CloudRain, 
  TrendingUp, 
  MapPin, 
  Calendar,
  Droplets,
  Thermometer,
  MessageSquare,
  User,
  LogOut,
  Home,
  BarChart3,
  Settings,
  HelpCircle,
  Wheat,
  Globe
} from 'lucide-react';

// Import Recharts for data visualization
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
const API = `${BACKEND_URL}/api`;

// Debug logging
console.log('BACKEND_URL:', BACKEND_URL);
console.log('API:', API);

// Auth Context
const AuthContext = React.createContext();

const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      fetchUser();
    } else {
      setLoading(false);
    }
  }, [token]);

  const fetchUser = async () => {
    try {
      const response = await axios.get(`${API}/auth/me`);
      setUser(response.data);
    } catch (error) {
      localStorage.removeItem('token');
      setToken(null);
      delete axios.defaults.headers.common['Authorization'];
    } finally {
      setLoading(false);
    }
  };

  const login = async (email, password) => {
    const response = await axios.post(`${API}/auth/login`, { email, password });
    const { access_token } = response.data;
    localStorage.setItem('token', access_token);
    setToken(access_token);
    axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
    return response.data;
  };

  const register = async (email, password, name, phone) => {
    const response = await axios.post(`${API}/auth/register`, { 
      email, 
      password, 
      name, 
      phone 
    });
    const { access_token } = response.data;
    localStorage.setItem('token', access_token);
    setToken(access_token);
    axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
    return response.data;
  };

  const logout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
    delete axios.defaults.headers.common['Authorization'];
  };

  return (
    <AuthContext.Provider value={{ user, login, register, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

const useAuth = () => {
  return React.useContext(AuthContext);
};

// Login Component
const LoginForm = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: '',
    phone: ''
  });
  const [loading, setLoading] = useState(false);
  const { login, register } = useAuth();
  const { toast } = useToast();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      if (isLogin) {
        await login(formData.email, formData.password);
        toast({
          title: "Success",
          description: "Logged in successfully!",
        });
      } else {
        await register(formData.email, formData.password, formData.name, formData.phone);
        toast({
          title: "Success",
          description: "Account created successfully!",
        });
      }
    } catch (error) {
      toast({
        title: "Error",
        description: error.response?.data?.detail || "An error occurred",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-green-50 via-yellow-50 to-orange-50">
      <div className="w-full max-w-md p-8">
        <Card className="shadow-2xl border-0 bg-white/95 backdrop-blur-sm">
          <CardHeader className="text-center pb-2">
            <div className="mx-auto mb-4 w-16 h-16 bg-gradient-to-br from-green-600 to-yellow-600 rounded-full flex items-center justify-center">
              <Sprout className="w-8 h-8 text-white" />
            </div>
            <CardTitle className="text-2xl font-bold text-green-800">KrishiSetu</CardTitle>
            <CardDescription className="text-green-600">
              AI-powered Crop Yield Prediction for Indian Farmers
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {!isLogin && (
                <>
                  <div className="space-y-2">
                    <Label htmlFor="name">Full Name</Label>
                    <Input
                      id="name"
                      type="text"
                      value={formData.name}
                      onChange={(e) => setFormData({...formData, name: e.target.value})}
                      required={!isLogin}
                      className="border-green-200 focus:border-green-500"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="phone">Phone Number</Label>
                    <Input
                      id="phone"
                      type="tel"
                      value={formData.phone}
                      onChange={(e) => setFormData({...formData, phone: e.target.value})}
                      className="border-green-200 focus:border-green-500"
                    />
                  </div>
                </>
              )}
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({...formData, email: e.target.value})}
                  required
                  className="border-green-200 focus:border-green-500"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  value={formData.password}
                  onChange={(e) => setFormData({...formData, password: e.target.value})}
                  required
                  className="border-green-200 focus:border-green-500"
                />
              </div>
              <Button 
                type="submit" 
                className="w-full bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white font-medium py-3"
                disabled={loading}
              >
                {loading ? 'Please wait...' : (isLogin ? 'Login' : 'Create Account')}
              </Button>
            </form>
            <div className="mt-4 text-center">
              <button
                type="button"
                onClick={() => setIsLogin(!isLogin)}
                className="text-green-600 hover:text-green-800 text-sm font-medium"
              >
                {isLogin ? "Don't have an account? Sign up" : "Already have an account? Login"}
              </button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

// Navigation Component
const Navigation = ({ activeTab, setActiveTab }) => {
  const { user, logout } = useAuth();

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Home },
    { id: 'predict', label: 'Predict Yield', icon: Sprout },
    { id: 'weather', label: 'Weather', icon: CloudRain },
    { id: 'chat', label: 'Assistant', icon: MessageSquare },
  ];

  return (
    <div className="bg-white shadow-lg border-b border-green-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-8">
            <div className="flex items-center space-x-2">
              <div className="w-10 h-10 bg-gradient-to-br from-green-600 to-yellow-600 rounded-lg flex items-center justify-center">
                <Sprout className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold text-green-800">KrishiSetu</span>
            </div>
            <nav className="hidden md:flex space-x-8">
              {navItems.map((item) => {
                const Icon = item.icon;
                return (
                  <button
                    key={item.id}
                    onClick={() => setActiveTab(item.id)}
                    className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                      activeTab === item.id
                        ? 'bg-green-100 text-green-800'
                        : 'text-gray-600 hover:text-green-800 hover:bg-green-50'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <User className="w-4 h-4 text-gray-600" />
              <span className="text-sm text-gray-700">{user?.name}</span>
            </div>
            <Button
              onClick={logout}
              variant="outline"
              size="sm"
              className="border-red-200 text-red-600 hover:bg-red-50 hover:border-red-300"
            >
              <LogOut className="w-4 h-4 mr-2" />
              Logout
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Dashboard Component  
const Dashboard = () => {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    fetchPredictions();
  }, []);

  const fetchPredictions = async () => {
    try {
      const response = await axios.get(`${API}/my-predictions`);
      setPredictions(response.data.predictions);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to fetch predictions",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  // Sample data for charts
  const yieldData = [
    { month: 'Jan', predicted: 12, actual: 11 },
    { month: 'Feb', predicted: 15, actual: 14 },
    { month: 'Mar', predicted: 18, actual: 17 },
    { month: 'Apr', predicted: 22, actual: 20 },
    { month: 'May', predicted: 16, actual: 18 },
    { month: 'Jun', predicted: 14, actual: 15 },
  ];

  const cropDistribution = [
    { name: 'Rice', value: 35, color: '#22c55e' },
    { name: 'Wheat', value: 25, color: '#eab308' },
    { name: 'Maize', value: 20, color: '#f97316' },
    { name: 'Cotton', value: 20, color: '#06b6d4' },
  ];

  if (loading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-gray-200 rounded w-1/4"></div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-32 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-green-800">Dashboard</h1>
        <Badge variant="secondary" className="bg-green-100 text-green-800">
          {predictions.length} Predictions Made
        </Badge>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="border-green-200">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Average Yield</p>
                <p className="text-2xl font-bold text-green-800">16.8 Q/Ha</p>
              </div>
              <TrendingUp className="w-8 h-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-blue-200">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Farm Area</p>
                <p className="text-2xl font-bold text-blue-800">4.2 Ha</p>
              </div>
              <MapPin className="w-8 h-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-orange-200">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Temperature</p>
                <p className="text-2xl font-bold text-orange-800">28°C</p>
              </div>
              <Thermometer className="w-8 h-8 text-orange-600" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-cyan-200">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Rainfall</p>
                <p className="text-2xl font-bold text-cyan-800">145 mm</p>
              </div>
              <Droplets className="w-8 h-8 text-cyan-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-green-800">Yield Trends</CardTitle>
            <CardDescription>Monthly predicted vs actual yield</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={yieldData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="predicted" 
                  stroke="#22c55e" 
                  strokeWidth={2}
                  name="Predicted"
                />
                <Line 
                  type="monotone" 
                  dataKey="actual" 
                  stroke="#f97316" 
                  strokeWidth={2}
                  name="Actual"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-green-800">Crop Distribution</CardTitle>
            <CardDescription>Your farm crop allocation</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={cropDistribution}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}%`}
                >
                  {cropDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Recent Predictions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-green-800">Recent Predictions</CardTitle>
          <CardDescription>Your latest crop yield predictions</CardDescription>
        </CardHeader>
        <CardContent>
          {predictions.length === 0 ? (
            <div className="text-center py-8">
              <Wheat className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">No predictions yet. Start by making your first prediction!</p>
            </div>
          ) : (
            <div className="space-y-4">
              {predictions.slice(0, 5).map((prediction, index) => (
                <div key={index} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                  <div className="flex items-center space-x-4">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <div>
                      <p className="font-medium text-gray-900">
                        {prediction.input_data?.crop_info?.crop_name || 'Unknown Crop'}
                      </p>
                      <p className="text-sm text-gray-600">
                        {new Date(prediction.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-bold text-green-800">
                      {prediction.predicted_yield} {prediction.yield_unit}
                    </p>
                    <p className="text-sm text-gray-600">
                      {prediction.comparison_percentage > 0 ? '+' : ''}{prediction.comparison_percentage}% vs district avg
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

// Crop Prediction Form Component
const CropPredictionForm = () => {
  const [formData, setFormData] = useState({
    // Farm Details
    state: '',
    district: '',
    village: '',
    pincode: '',
    farm_size: '',
    farm_size_unit: 'hectare',
    
    // Crop Info
    crop_name: '',
    variety: '',
    sowing_date: '',
    season: '',
    
    // Soil & Inputs
    soil_type: '',
    fertilizer_used: '',
    ph_level: '',
    organic_carbon: '',
    
    // Irrigation
    irrigation_source: '',
    irrigation_frequency: '',
    water_availability: ''
  });

  const [states, setStates] = useState([]);
  const [districts, setDistricts] = useState([]);
  const [crops, setCrops] = useState([]);
  const [soilTypes, setSoilTypes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const { toast } = useToast();
  const { user } = useAuth();

  useEffect(() => {
    fetchStates();
    fetchCrops();
    fetchSoilTypes();
  }, []);

  useEffect(() => {
    if (formData.state) {
      fetchDistricts(formData.state);
    }
  }, [formData.state]);

  const fetchStates = async () => {
    try {
      const response = await axios.get(`${API}/states`);
      setStates(response.data.states);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to fetch states",
        variant: "destructive",
      });
    }
  };

  const fetchDistricts = async (state) => {
    try {
      const response = await axios.get(`${API}/districts/${state}`);
      setDistricts(response.data.districts);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to fetch districts",
        variant: "destructive",
      });
    }
  };

  const fetchCrops = async () => {
    try {
      const response = await axios.get(`${API}/crops`);
      setCrops(response.data.crops);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to fetch crops",
        variant: "destructive",
      });
    }
  };

  const fetchSoilTypes = async () => {
    try {
      const response = await axios.get(`${API}/soil-types`);
      setSoilTypes(response.data.soilTypes);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to fetch soil types",
        variant: "destructive",
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const requestData = {
        user_id: user.id,
        farm_details: {
          state: formData.state,
          district: formData.district,
          village: formData.village,
          pincode: formData.pincode,
          farm_size: parseFloat(formData.farm_size),
          farm_size_unit: formData.farm_size_unit
        },
        crop_info: {
          crop_name: formData.crop_name,
          variety: formData.variety,
          sowing_date: formData.sowing_date,
          season: formData.season
        },
        soil_inputs: {
          soil_type: formData.soil_type,
          fertilizer_used: formData.fertilizer_used,
          ph_level: formData.ph_level ? parseFloat(formData.ph_level) : null,
          organic_carbon: formData.organic_carbon ? parseFloat(formData.organic_carbon) : null
        },
        irrigation_info: {
          irrigation_source: formData.irrigation_source,
          irrigation_frequency: formData.irrigation_frequency,
          water_availability: formData.water_availability
        }
      };

      const response = await axios.post(`${API}/predict-yield`, requestData);
      setPrediction(response.data);
      
      toast({
        title: "Success",
        description: "Crop yield prediction generated successfully!",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: error.response?.data?.detail || "Failed to generate prediction",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  if (prediction) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold text-green-800">Prediction Result</h1>
          <Button 
            onClick={() => setPrediction(null)}
            variant="outline"
            className="border-green-200 text-green-600 hover:bg-green-50"
          >
            Make Another Prediction
          </Button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Main Result Card */}
          <Card className="lg:col-span-2 border-green-200">
            <CardHeader className="bg-gradient-to-r from-green-50 to-yellow-50">
              <CardTitle className="text-2xl text-green-800 flex items-center">
                <TrendingUp className="w-6 h-6 mr-2" />
                Predicted Yield
              </CardTitle>
            </CardHeader>
            <CardContent className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <p className="text-sm text-gray-600 mb-2">Your Predicted Yield</p>
                  <p className="text-4xl font-bold text-green-800">
                    {prediction.predicted_yield}
                  </p>
                  <p className="text-sm text-gray-600">{prediction.yield_unit}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-600 mb-2">District Average</p>
                  <p className="text-4xl font-bold text-gray-700">
                    {prediction.district_average}
                  </p>
                  <p className="text-sm text-gray-600">{prediction.yield_unit}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-600 mb-2">Comparison</p>
                  <p className={`text-4xl font-bold ${prediction.comparison_percentage >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {prediction.comparison_percentage >= 0 ? '+' : ''}{prediction.comparison_percentage}%
                  </p>
                  <p className="text-sm text-gray-600">vs district avg</p>
                </div>
              </div>
              <div className="mt-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600">Confidence Score</span>
                  <span className="text-sm font-medium text-gray-800">{prediction.confidence_score}%</span>
                </div>
                <Progress value={prediction.confidence_score} className="h-3" />
              </div>
            </CardContent>
          </Card>

          {/* Recommendations Card */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="text-green-800 flex items-center">
                <HelpCircle className="w-5 h-5 mr-2" />
                Recommendations
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {prediction.recommendations.map((recommendation, index) => (
                  <Alert key={index} className="border-green-200">
                    <HelpCircle className="h-4 w-4" />
                    <AlertDescription className="text-green-800">
                      {recommendation}
                    </AlertDescription>
                  </Alert>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-green-800">Crop Yield Prediction</h1>
        <Badge variant="secondary" className="bg-blue-100 text-blue-800">
          AI-Powered Analysis
        </Badge>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Farm Details Section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-green-800 flex items-center">
              <MapPin className="w-5 h-5 mr-2" />
              Farm Details
            </CardTitle>
            <CardDescription>
              Basic information about your farm location and size
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="state">State *</Label>
                <Select value={formData.state} onValueChange={(value) => handleInputChange('state', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select your state" />
                  </SelectTrigger>
                  <SelectContent>
                    {states.map((state) => (
                      <SelectItem key={state} value={state}>{state}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="district">District *</Label>
                <Select 
                  value={formData.district} 
                  onValueChange={(value) => handleInputChange('district', value)}
                  disabled={!formData.state}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select your district" />
                  </SelectTrigger>
                  <SelectContent>
                    {districts.map((district) => (
                      <SelectItem key={district} value={district}>{district}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="village">Village/Area</Label>
                <Input
                  id="village"
                  value={formData.village}
                  onChange={(e) => handleInputChange('village', e.target.value)}
                  placeholder="Enter village name"
                  className="border-green-200 focus:border-green-500"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="pincode">Pincode</Label>
                <Input
                  id="pincode"
                  value={formData.pincode}
                  onChange={(e) => handleInputChange('pincode', e.target.value)}
                  placeholder="Enter pincode"
                  className="border-green-200 focus:border-green-500"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="farm_size">Farm Size *</Label>
                <Input
                  id="farm_size"
                  type="number"
                  step="0.1"
                  value={formData.farm_size}
                  onChange={(e) => handleInputChange('farm_size', e.target.value)}
                  placeholder="Enter farm size"
                  className="border-green-200 focus:border-green-500"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="farm_size_unit">Unit</Label>
                <Select value={formData.farm_size_unit} onValueChange={(value) => handleInputChange('farm_size_unit', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="acre">Acre</SelectItem>
                    <SelectItem value="hectare">Hectare</SelectItem>
                    <SelectItem value="bigha">Bigha</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Crop Information Section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-green-800 flex items-center">
              <Wheat className="w-5 h-5 mr-2" />
              Crop Information
            </CardTitle>
            <CardDescription>
              Details about the crop you want to predict
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="crop_name">Crop Name *</Label>
                <Select value={formData.crop_name} onValueChange={(value) => handleInputChange('crop_name', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select crop" />
                  </SelectTrigger>
                  <SelectContent>
                    {crops.map((crop) => (
                      <SelectItem key={crop} value={crop}>{crop}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="variety">Variety/Seed Type</Label>
                <Select value={formData.variety} onValueChange={(value) => handleInputChange('variety', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select variety" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Local">Local</SelectItem>
                    <SelectItem value="Hybrid">Hybrid</SelectItem>
                    <SelectItem value="HYV">High Yielding Variety</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="sowing_date">Sowing Date</Label>
                <Input
                  id="sowing_date"
                  type="date"
                  value={formData.sowing_date}
                  onChange={(e) => handleInputChange('sowing_date', e.target.value)}
                  className="border-green-200 focus:border-green-500"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="season">Season *</Label>
                <Select value={formData.season} onValueChange={(value) => handleInputChange('season', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select season" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Kharif">Kharif (Monsoon)</SelectItem>
                    <SelectItem value="Rabi">Rabi (Winter)</SelectItem>
                    <SelectItem value="Zaid">Zaid (Summer)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Soil & Inputs Section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-green-800 flex items-center">
              <Globe className="w-5 h-5 mr-2" />
              Soil & Inputs
            </CardTitle>
            <CardDescription>
              Information about your soil type and fertilizer usage
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="soil_type">Soil Type *</Label>
                <Select value={formData.soil_type} onValueChange={(value) => handleInputChange('soil_type', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select soil type" />
                  </SelectTrigger>
                  <SelectContent>
                    {soilTypes.map((soil) => (
                      <SelectItem key={soil.name} value={soil.name}>
                        {soil.name} - {soil.description}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="fertilizer_used">Fertilizer Used</Label>
                <Select value={formData.fertilizer_used} onValueChange={(value) => handleInputChange('fertilizer_used', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select fertilizer type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="None">None</SelectItem>
                    <SelectItem value="Organic">Organic</SelectItem>
                    <SelectItem value="Urea+DAP">Urea + DAP</SelectItem>
                    <SelectItem value="Mixed">Mixed</SelectItem>
                    <SelectItem value="NPK">NPK Complex</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="ph_level">Soil pH Level (Optional)</Label>
                <Input
                  id="ph_level"
                  type="number"
                  step="0.1"
                  min="0"
                  max="14"
                  value={formData.ph_level}
                  onChange={(e) => handleInputChange('ph_level', e.target.value)}
                  placeholder="e.g., 6.5"
                  className="border-green-200 focus:border-green-500"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="organic_carbon">Organic Carbon % (Optional)</Label>
                <Input
                  id="organic_carbon"
                  type="number"
                  step="0.01"
                  value={formData.organic_carbon}
                  onChange={(e) => handleInputChange('organic_carbon', e.target.value)}
                  placeholder="e.g., 0.75"
                  className="border-green-200 focus:border-green-500"  
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Irrigation & Water Section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-green-800 flex items-center">
              <Droplets className="w-5 h-5 mr-2" />
              Irrigation & Water
            </CardTitle>
            <CardDescription>
              Water management and irrigation details
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="irrigation_source">Irrigation Source *</Label>
                <Select value={formData.irrigation_source} onValueChange={(value) => handleInputChange('irrigation_source', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select irrigation source" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Rainfed">Rainfed</SelectItem>
                    <SelectItem value="Borewell">Borewell</SelectItem>
                    <SelectItem value="Canal">Canal</SelectItem>
                    <SelectItem value="Tank/Pond">Tank/Pond</SelectItem>
                    <SelectItem value="River">River</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="irrigation_frequency">Irrigation Frequency *</Label>
                <Select value={formData.irrigation_frequency} onValueChange={(value) => handleInputChange('irrigation_frequency', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select frequency" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Rarely">Rarely</SelectItem>
                    <SelectItem value="Sometimes">Sometimes</SelectItem>
                    <SelectItem value="Regularly">Regularly</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2 md:col-span-2">
                <Label htmlFor="water_availability">Water Availability</Label>
                <Select value={formData.water_availability} onValueChange={(value) => handleInputChange('water_availability', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select water availability" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Abundant">Abundant</SelectItem>
                    <SelectItem value="Adequate">Adequate</SelectItem>
                    <SelectItem value="Limited">Limited</SelectItem>
                    <SelectItem value="Scarce">Scarce</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Submit Button */}
        <div className="flex justify-center">
          <Button 
            type="submit" 
            size="lg"
            className="bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white px-8 py-3 text-lg font-medium"
            disabled={loading || !formData.state || !formData.crop_name || !formData.season || !formData.soil_type || !formData.irrigation_source || !formData.irrigation_frequency || !formData.farm_size}
          >
            {loading ? 'Analyzing...' : 'Predict Yield'}
            <TrendingUp className="w-5 h-5 ml-2" />
          </Button>
        </div>
      </form>
    </div>
  );
};

// Weather Component
const WeatherDashboard = () => {
  const [weatherData, setWeatherData] = useState(null);
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    // Get weather for a default location (Delhi)
    fetchWeather(28.6139, 77.2090);
  }, []);

  const fetchWeather = async (lat, lon) => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/weather/${lat}/${lon}`);
      setWeatherData(response.data);
    } catch (error) {
      // Use mock data if API fails
      setWeatherData({
        temperature: 28.5,
        humidity: 65,
        description: "partly cloudy",
        rainfall: 2.5
      });
      toast({
        title: "Info",
        description: "Using sample weather data",
      });
    } finally {
      setLoading(false);
    }
  };

  const weatherRecommendations = [
    {
      title: "Irrigation Schedule",
      description: "Based on current weather, irrigate crops in early morning hours.",
      icon: Droplets,
      color: "blue"
    },
    {
      title: "Temperature Alert",
      description: "Moderate temperature is good for most crops. Monitor for heat stress.",
      icon: Thermometer,
      color: "orange"
    },
    {
      title: "Humidity Levels",
      description: "Current humidity is optimal for plant growth.",
      icon: CloudRain,
      color: "green"
    }
  ];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-green-800">Weather Dashboard</h1>
        <Badge variant="secondary" className="bg-blue-100 text-blue-800">
          Real-time Data
        </Badge>
      </div>

      {loading ? (
        <div className="animate-pulse space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-32 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      ) : (
        <>
          {/* Weather Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card className="border-orange-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Temperature</p>
                    <p className="text-3xl font-bold text-orange-800">
                      {weatherData?.temperature || 0}°C
                    </p>
                  </div>
                  <Thermometer className="w-8 h-8 text-orange-600" />
                </div>
              </CardContent>
            </Card>

            <Card className="border-blue-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Humidity</p>
                    <p className="text-3xl font-bold text-blue-800">
                      {weatherData?.humidity || 0}%
                    </p>
                  </div>
                  <Droplets className="w-8 h-8 text-blue-600" />
                </div>
              </CardContent>
            </Card>

            <Card className="border-cyan-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Rainfall</p>
                    <p className="text-3xl font-bold text-cyan-800">
                      {weatherData?.rainfall || 0} mm
                    </p>
                  </div>
                  <CloudRain className="w-8 h-8 text-cyan-600" />
                </div>
              </CardContent>
            </Card>

            <Card className="border-green-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Condition</p>
                    <p className="text-lg font-bold text-green-800 capitalize">
                      {weatherData?.description || 'Clear'}
                    </p>
                  </div>
                  <CloudRain className="w-8 h-8 text-green-600" />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Weather-based Recommendations */}
          <Card>
            <CardHeader>
              <CardTitle className="text-green-800">Weather-based Recommendations</CardTitle>
              <CardDescription>
                Actionable advice based on current weather conditions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {weatherRecommendations.map((rec, index) => {
                  const Icon = rec.icon;
                  return (
                    <div key={index} className="flex items-start space-x-3 p-4 border rounded-lg">
                      <div className={`p-2 rounded-lg bg-${rec.color}-100`}>
                        <Icon className={`w-5 h-5 text-${rec.color}-600`} />
                      </div>
                      <div>
                        <h3 className="font-medium text-gray-900">{rec.title}</h3>
                        <p className="text-sm text-gray-600 mt-1">{rec.description}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
};

// Chat Assistant Component
const ChatAssistant = () => {
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      content: 'Hello! I\'m your farming assistant. How can I help you today?',
      language: 'en'
    }
  ]);
  const [newMessage, setNewMessage] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  const languages = [
    { code: 'en', name: 'English', flag: '🇺🇸' },
    { code: 'hi', name: 'हिंदी', flag: '🇮🇳' },
    { code: 'bn', name: 'বাংলা', flag: '🇧🇩' },
    { code: 'or', name: 'ଓଡ଼ିଆ', flag: '🇮🇳' }
  ];

  const sendMessage = async () => {
    if (!newMessage.trim()) return;

    const userMessage = {
      type: 'user',
      content: newMessage,
      language: selectedLanguage
    };

    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    try {
      const response = await axios.post(`${API}/chat`, {
        message: newMessage,
        language: selectedLanguage
      });

      const botMessage = {
        type: 'bot',
        content: response.data.response,
        language: selectedLanguage,
        recommendations: response.data.recommendations
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to get response from assistant",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
      setNewMessage('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-green-800">Farming Assistant</h1>
        <div className="flex items-center space-x-2">
          <Label htmlFor="language">Language:</Label>
          <Select value={selectedLanguage} onValueChange={setSelectedLanguage}>
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {languages.map((lang) => (
                <SelectItem key={lang.code} value={lang.code}>
                  {lang.flag} {lang.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <Card className="h-96">
        <CardContent className="p-0 h-full flex flex-col">
          {/* Messages */}
          <div className="flex-1 p-4 overflow-y-auto space-y-4">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                    msg.type === 'user'
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  <p className="text-sm">{msg.content}</p>
                  {msg.recommendations && (
                    <div className="mt-2 space-y-1">
                      {msg.recommendations.map((rec, idx) => (
                        <p key={idx} className="text-xs opacity-75">• {rec}</p>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 px-4 py-2 rounded-lg">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Input */}
          <div className="p-4 border-t border-gray-200">
            <div className="flex space-x-2">
              <Textarea
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me anything about farming..."
                className="flex-1 min-h-[40px] max-h-[120px] border-green-200 focus:border-green-500"
                disabled={loading}
              />
              <Button
                onClick={sendMessage}
                disabled={loading || !newMessage.trim()}
                className="bg-green-600 hover:bg-green-700 text-white"
              >
                <MessageSquare className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-green-800">Quick Questions</CardTitle>
          <CardDescription>
            Click on these common questions to get instant answers
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {[
              "How often should I water my crops?",
              "What fertilizer is best for rice?",
              "How to control pests naturally?",
              "When to harvest wheat?",
              "Best irrigation practices",
              "Soil pH management"
            ].map((question, index) => (
              <Button
                key={index}
                variant="outline"
                onClick={() => setNewMessage(question)}
                className="text-left justify-start border-green-200 hover:bg-green-50 hover:border-green-300"
              >
                {question}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Main App Component
const MainApp = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const { user, loading } = useAuth();

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard />;
      case 'predict':
        return <CropPredictionForm />;
      case 'weather':
        return <WeatherDashboard />;
      case 'chat':
        return <ChatAssistant />;
      default:
        return <Dashboard />;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-green-50 to-yellow-50">
        <div className="text-center">
          <div className="w-16 h-16 bg-gradient-to-br from-green-600 to-yellow-600 rounded-full flex items-center justify-center mx-auto mb-4 animate-pulse">
            <Sprout className="w-8 h-8 text-white" />
          </div>
          <p className="text-green-800 font-medium">Loading KrishiSetu...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return <LoginForm />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-yellow-50 to-orange-50">
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="max-w-7xl mx-auto">
        {renderContent()}
      </main>
      <Toaster />
    </div>
  );
};

// App Component with Router
function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/*" element={<MainApp />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;