# from sklearn.ensemble import ExtraTreeRegressor,RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import Ridge,Lasso,LinearRegression
import warnings


print("-"*67)