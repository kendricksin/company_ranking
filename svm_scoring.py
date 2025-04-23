#!/usr/bin/env python3
"""
Enhanced Gradient Descent Model for Competitor Scoring with Direct Database Connection

This script implements an enhanced version of the gradient descent model
that connects directly to the PostgreSQL database, with additional features
and fine-tuning based on domain knowledge.
"""

import os
import sys
import asyncio
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # Import SVM classifier
from tabulate import tabulate
import logging
import asyncpg
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class DbConnection:
    """Database connection class with enhanced error handling."""
    def __init__(self):
        self.pool = None

    async def initialize(self):
        """Initialize connection pool with retry logic"""
        # Get database connection parameters from environment variables
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = int(os.getenv("DB_PORT", "5432"))
        db_name = os.getenv("DB_NAME", "postgres")
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "postgres")

        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Create connection pool
                self.pool = await asyncpg.create_pool(
                    host=db_host,
                    port=db_port,
                    database=db_name,
                    user=db_user,
                    password=db_password,
                    min_size=5,
                    max_size=20
                )

                if self.pool:
                    logger.info(f"Successfully connected to database {db_name} at {db_host}:{db_port}")
                    return
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Database connection attempt {retry_count} failed: {str(e)}")
                await asyncio.sleep(1) # Wait before retrying
        logger.critical("Failed to create database connection after multiple attempts")
        sys.exit(1)

    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    async def read_sql_async(self, query: str, params: list = None) -> pd.DataFrame:
        """Execute a query and return the results as a pandas DataFrame with error handling"""
        if not self.pool:
            raise Exception("Database connection not initialized")

        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetch(query, *(params or []))

                # Convert to pandas DataFrame
                if result:
                    return pd.DataFrame(result, columns=[k for k in result[0].keys()])
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Database query error: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            # Return empty DataFrame instead of raising to allow partial processing
            return pd.DataFrame()

class SVMClassifier:
    """Support Vector Machine Classifier wrapper."""
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', class_weight='balanced'):
        self.C = C  # Regularization parameter
        self.kernel = kernel  # Kernel type
        self.gamma = gamma  # Kernel coefficient
        self.class_weight = class_weight  # Class weight
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            class_weight=self.class_weight,
            probability=True  # Enable probability estimates
        )
        self.scaler = StandardScaler()
        self.feature_names = None

    def fit(self, X, y, feature_names=None):
        n_samples, n_features = X.shape
        self.feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(n_features)]

        # Feature Scaling
        X_scaled = self.scaler.fit_transform(X)

        # Train SVM model
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        """Predicts probabilities."""
        if self.model is None:
            raise Exception("Model has not been trained yet.")

        X_scaled = self.scaler.transform(X)
        # Return probabilities for the positive class (class 1)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X, threshold=0.5):
        """Predicts class labels based on threshold."""
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= threshold, 1, 0)

    def get_feature_weights(self):
        """Returns a dictionary of feature names and their importance scores."""
        if self.model is None or self.feature_names is None:
            raise Exception("Model has not been trained yet or feature names not set.")
            
        # For linear kernel, we can use coefficients directly
        if self.kernel == 'linear':
            weights = self.model.coef_[0]
            return dict(zip(self.feature_names, weights))
        
        # For non-linear kernels, use coefficients of the support vectors
        # This is an approximation of feature importance
        else:
            # Get absolute sum of dual coefficients for each support vector
            sv_coef_abs_sum = np.abs(self.model.dual_coef_).sum(axis=0)
            
            # For each feature, calculate importance as weighted sum of support vector values
            feature_weights = {}
            X_support = self.scaler.inverse_transform(self.model.support_vectors_)
            
            for i, feature_name in enumerate(self.feature_names):
                importance = np.sum(X_support[:, i] * sv_coef_abs_sum)
                # Normalize to have similar scale as gradient descent weights
                feature_weights[feature_name] = importance / np.sum(np.abs(importance)) 
            
            return feature_weights

class SimpleGradientDescentClassifier:
    """Simple Logistic Regression Classifier trained with Gradient Descent."""
    def __init__(self, learning_rate=0.01, iterations=1000, regularization=0.01):
        self.lr = learning_rate
        self.iter = iterations
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def sigmoid(self, z):
        # Clip input to avoid overflow in exp
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def fit(self, X, y, feature_names=None):
        n_samples, n_features = X.shape
        self.feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(n_features)]

        # Feature Scaling
        X_scaled = self.scaler.fit_transform(X)

        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.iter):
            linear_model = np.dot(X_scaled, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Compute gradients with L2 regularization
            dw = (1/n_samples) * (np.dot(X_scaled.T, (y_pred - y)) + self.regularization * self.weights)
            db = (1/n_samples) * np.sum(y_pred - y)

            # Update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        """Predicts probabilities (sigmoid output)."""
        if self.weights is None or self.bias is None:
            raise Exception("Model has not been trained yet.")

        X_scaled = self.scaler.transform(X)
        linear_model = np.dot(X_scaled, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """Predicts class labels based on threshold."""
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= threshold, 1, 0)

    def get_feature_weights(self):
        """Returns a dictionary of feature names and their weights."""
        if self.weights is None or self.feature_names is None:
            raise Exception("Model has not been trained yet or feature names not set.")
            
        return dict(zip(self.feature_names, self.weights))


class DatabaseGradientModel:
    """Class for building and analyzing a gradient model with direct database access."""
    
    def __init__(self, db_conn):
        """Initialize with database connection."""
        self.db = db_conn
        self.data = None
        self.model = None
        
        # Configure known strong competitors and target company
        self.known_strong_competitors = [
            "0105540085581",  # บริษัท ตรีสกุล จำกัด
            "0103503000975",  # ห้างหุ้นส่วนจำกัด แสงนิยม
            "0105538044211",  # บริษัท ปาล์ม คอน จำกัด
        ]
        self.target_company_tin = "0105543041542"  # บริษัท เรืองฤทัย จำกัด
        
        # Set output directory
        self.output_dir = 'gradient_output'
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def get_target_company_info(self):
        """Get detailed information about the target company."""
        logger.info(f"Fetching company info for target: {self.target_company_tin}")
        
        query = """
        SELECT
            b.tin,
            b.company as name,
            COUNT(DISTINCT b.project_id) as total_projects,
            SUM(CASE WHEN p.winner_tin = b.tin THEN 1 ELSE 0 END) as total_wins,
            AVG(b.bid) as avg_bid_amount,
            AVG(p.sum_price_agree) as avg_project_value,
            COUNT(DISTINCT p.dept_name) as unique_departments,
            COUNT(DISTINCT p.project_type_name) as unique_project_types,
            MIN(p.contract_date) as first_project_date,
            MAX(p.contract_date) as last_project_date
        FROM public_data.thai_project_bid_info b
        JOIN public_data.thai_govt_project p ON b.project_id = p.project_id
        WHERE b.tin = $1
        GROUP BY b.tin, b.company
        """
        
        company_df = await self.db.read_sql_async(query, [self.target_company_tin])
        
        if company_df.empty:
            logger.warning(f"No company information found for target: {self.target_company_tin}")
            return None
        
        company_data = company_df.iloc[0].to_dict()
        
        # Calculate additional derived metrics
        company_data['win_rate'] = (
            (company_data['total_wins'] / company_data['total_projects']) * 100
        ) if company_data['total_projects'] > 0 else 0.0
        
        # Ensure unique_departments exists and is > 0
        company_data['unique_departments'] = company_data.get('unique_departments', 0)
        if company_data['unique_departments'] is None or company_data['unique_departments'] == 0:
            company_data['unique_departments'] = 1  # Avoid division by zero
        
        logger.info(f"Target company: {company_data['name']} ({self.target_company_tin})")
        return company_data
    
    async def get_all_competitors(self, lookback_months=60):
        """Get all competitors for a target company from the database."""
        logger.info(f"Fetching all competitors for company TIN: {self.target_company_tin}")
        
        time_limit_condition = ""
        params = [self.target_company_tin]
        
        if lookback_months > 0:
            time_limit_condition = f"AND p.contract_date >= NOW() - INTERVAL '{lookback_months} months'"
        
        competitors_query = f"""
        -- First, identify competitors with direct project overlap
        WITH direct_competitors AS (
            SELECT
                b.tin,
                MAX(b.company) as name, -- Handle different names for same TIN
                COUNT(DISTINCT b.project_id) as common_projects
            FROM public_data.thai_project_bid_info b
            JOIN public_data.thai_govt_project p ON b.project_id = p.project_id -- Need p for time limit
            JOIN (
                -- Get all projects for the target company within the lookback period
                SELECT DISTINCT b_sub.project_id -- Specify b_sub.project_id
                FROM public_data.thai_project_bid_info b_sub
                JOIN public_data.thai_govt_project p_sub ON b_sub.project_id = p_sub.project_id
                WHERE b_sub.tin = $1 {time_limit_condition.replace('p.contract_date', 'p_sub.contract_date')} -- Apply time limit here, adjust alias
            ) cp ON b.project_id = cp.project_id -- Join based on common project IDs
            WHERE b.tin != $1 {time_limit_condition} -- Apply time limit to competitor's projects too
            GROUP BY b.tin
            HAVING COUNT(DISTINCT b.project_id) >= 1
        ),
        -- Get departments where the target company operates within the lookback period
        company_departments AS (
            SELECT DISTINCT p.dept_name
            FROM public_data.thai_project_bid_info b
            JOIN public_data.thai_govt_project p ON b.project_id = p.project_id
            WHERE b.tin = $1 {time_limit_condition}
        ),
        -- Get comprehensive stats for identified competitors within the lookback period
        competitor_stats AS (
            SELECT
                dc.tin,
                dc.name,
                dc.common_projects, -- Already filtered by time via direct_competitors CTE
                -- Total projects within the lookback period
                (
                    SELECT COUNT(DISTINCT b.project_id)
                    FROM public_data.thai_project_bid_info b
                    JOIN public_data.thai_govt_project p ON b.project_id = p.project_id
                    WHERE b.tin = dc.tin {time_limit_condition}
                ) as total_projects,
                -- Total wins within the lookback period
                (
                    SELECT COUNT(DISTINCT p.project_id) -- Can use p.project_id or b.project_id here
                    FROM public_data.thai_project_bid_info b
                    JOIN public_data.thai_govt_project p ON b.project_id = p.project_id
                    WHERE b.tin = dc.tin AND p.winner_tin = b.tin {time_limit_condition}
                ) as total_wins,
                -- Common departments (count based on competitor's projects in target's departments)
                (
                    SELECT COUNT(DISTINCT p.dept_name)
                    FROM public_data.thai_project_bid_info b
                    JOIN public_data.thai_govt_project p ON b.project_id = p.project_id
                    JOIN company_departments cd ON p.dept_name = cd.dept_name -- cd is already time-filtered
                    WHERE b.tin = dc.tin {time_limit_condition} -- Filter competitor's projects by time too
                ) as common_departments,
                -- Wins in common departments within the lookback period
                (
                    SELECT COUNT(DISTINCT p.project_id) -- Can use p.project_id or b.project_id here
                    FROM public_data.thai_project_bid_info b
                    JOIN public_data.thai_govt_project p ON b.project_id = p.project_id
                    JOIN company_departments cd ON p.dept_name = cd.dept_name -- cd is already time-filtered
                    WHERE b.tin = dc.tin AND p.winner_tin = b.tin {time_limit_condition} -- Filter competitor's wins by time
                ) as dept_wins,
                -- Average bid amount within the lookback period
                (
                    SELECT AVG(b.bid)
                    FROM public_data.thai_project_bid_info b
                    JOIN public_data.thai_govt_project p ON b.project_id = p.project_id
                    WHERE b.tin = dc.tin {time_limit_condition}
                ) as avg_bid_amount,
                -- Most recent activity within the lookback period
                (
                    SELECT MAX(p.contract_date)
                    FROM public_data.thai_project_bid_info b
                    JOIN public_data.thai_govt_project p ON b.project_id = p.project_id
                    WHERE b.tin = dc.tin {time_limit_condition}
                ) as last_activity_date,
                -- Total contract value for won projects
                (
                    SELECT SUM(p.sum_price_agree)
                    FROM public_data.thai_project_bid_info b
                    JOIN public_data.thai_govt_project p ON b.project_id = p.project_id
                    WHERE b.tin = dc.tin AND p.winner_tin = b.tin {time_limit_condition}
                ) as total_contract_value,
                -- Common subdepartments
                (
                    SELECT COUNT(DISTINCT p.dept_sub_name)
                    FROM public_data.thai_project_bid_info b
                    JOIN public_data.thai_govt_project p ON b.project_id = p.project_id
                    JOIN (
                        SELECT DISTINCT p_sub.dept_sub_name
                        FROM public_data.thai_project_bid_info b_sub
                        JOIN public_data.thai_govt_project p_sub ON b_sub.project_id = p_sub.project_id
                        WHERE b_sub.tin = $1 {time_limit_condition.replace('p.contract_date', 'p_sub.contract_date')}
                    ) sd ON p.dept_sub_name = sd.dept_sub_name
                    WHERE b.tin = dc.tin {time_limit_condition}
                ) as common_subdepartments
            FROM direct_competitors dc
        )
        -- Apply final calculations and sorting
        SELECT
            tin,
            name,
            common_projects,
            common_departments,
            common_subdepartments,
            total_projects,
            total_wins,
            dept_wins,
            avg_bid_amount,
            last_activity_date,
            total_contract_value,
            -- Derived fields
            CASE WHEN total_projects > 0 THEN ROUND((total_wins::numeric / total_projects) * 100, 2) ELSE 0 END as win_rate,
            -- Calculate months since last activity relative to NOW()
            EXTRACT(DAYS FROM (NOW() - COALESCE(last_activity_date, NOW() - INTERVAL '{lookback_months + 1} months'))) / 30.0 as months_since_last_activity
        FROM competitor_stats
        -- Ensure total_projects is positive to avoid division by zero in score calculation and relevance filter
        WHERE total_projects > 0
        """
        
        competitors_df = await self.db.read_sql_async(competitors_query, params)
        
        # Filter out companies with TINs starting with "D88"
        original_count = len(competitors_df)
        competitors_df = competitors_df[~competitors_df['tin'].str.startswith('D88', na=False)]
        filtered_count = original_count - len(competitors_df)
        
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} companies with TIN starting with 'D88'")
        
        if competitors_df.empty:
            logger.warning(f"No competitors found for company TIN: {self.target_company_tin} within the lookback period.")
        else:
            logger.info(f"Found {len(competitors_df)} potential competitors after filtering")
        
        return competitors_df
    
    async def calculate_additional_metrics(self, competitors_df, target_company):
        """Calculate additional metrics for all competitors."""
        logger.info(f"Calculating additional metrics for {len(competitors_df)} competitors")
        
        if competitors_df.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        result_df = competitors_df.copy()
        
        # Calculate initial metrics
        dept_overlap_pct = (result_df['common_departments'] / target_company['unique_departments']) * 100
        result_df['dept_overlap_pct'] = dept_overlap_pct
        
        # Calculate price range overlap
        price_range_overlap = await self._calculate_price_range_overlap(
            result_df['tin'].tolist(), self.target_company_tin
        )
        result_df['price_range_overlap'] = price_range_overlap
        
        # Calculate recent activity score
        recent_activity = await self._calculate_recent_activity(
            result_df['tin'].tolist(), self.target_company_tin
        )
        result_df['recent_activity_score'] = recent_activity
        
        # Calculate total win value
        result_df['total_win_value'] = result_df['total_wins'] * result_df['avg_bid_amount']
        
        # Calculate derived features
        result_df['dept_engagement_score'] = (
            result_df['common_departments'] * 
            (result_df['dept_wins'] / result_df['common_departments'].clip(1))
        )
        
        result_df['subdept_specialization'] = (
            result_df['common_subdepartments'] / 
            result_df['common_departments'].clip(1)
        )
        
        result_df['win_density'] = result_df['total_wins'] / result_df['total_projects'].clip(1)
        
        # Fill missing values
        numeric_cols = result_df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if result_df[col].isnull().any():
                null_count = result_df[col].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Column '{col}' has {null_count} null values, filling with 0")
                    result_df[col] = result_df[col].fillna(0)
        
        return result_df
    
    async def _calculate_price_range_overlap(self, competitor_tins, target_tin):
        """Calculate price range overlap between target and competitors."""
        logger.info("Calculating price range overlap")
        
        # Get target price range
        target_range_query = """
        SELECT 
            MIN(p.sum_price_agree) as min_price,
            MAX(p.sum_price_agree) as max_price
        FROM public_data.thai_project_bid_info b
        JOIN public_data.thai_govt_project p ON b.project_id = p.project_id
        WHERE b.tin = $1 AND p.sum_price_agree > 0
        """
        
        target_range_df = await self.db.read_sql_async(target_range_query, [target_tin])
        
        if target_range_df.empty or pd.isna(target_range_df['min_price'].iloc[0]) or pd.isna(target_range_df['max_price'].iloc[0]):
            logger.warning("Could not calculate target price range")
            return pd.Series(0, index=np.arange(len(competitor_tins)))
        
        target_min = target_range_df['min_price'].iloc[0]
        target_max = target_range_df['max_price'].iloc[0]
        
        # Get competitor price ranges
        overlaps = []
        
        for tin in competitor_tins:
            comp_range_query = """
            SELECT 
                MIN(p.sum_price_agree) as min_price,
                MAX(p.sum_price_agree) as max_price
            FROM public_data.thai_project_bid_info b
            JOIN public_data.thai_govt_project p ON b.project_id = p.project_id
            WHERE b.tin = $1 AND p.sum_price_agree > 0
            """
            
            comp_range_df = await self.db.read_sql_async(comp_range_query, [tin])
            
            if comp_range_df.empty or pd.isna(comp_range_df['min_price'].iloc[0]) or pd.isna(comp_range_df['max_price'].iloc[0]):
                overlaps.append(0)
                continue
            
            comp_min = comp_range_df['min_price'].iloc[0]
            comp_max = comp_range_df['max_price'].iloc[0]
            
            # Calculate overlap
            overlap_start = max(target_min, comp_min)
            overlap_end = min(target_max, comp_max)
            
            if overlap_start >= overlap_end:
                overlaps.append(0)
            else:
                overlap_length = overlap_end - overlap_start
                target_length = target_max - target_min
                
                overlap_pct = (overlap_length / target_length) * 100 if target_length > 0 else 0
                overlaps.append(overlap_pct)
        
        return pd.Series(overlaps)
    
    async def _calculate_recent_activity(self, competitor_tins, target_tin, lookback_months=24):
        """Calculate how recently the companies have competed against each other."""
        logger.info("Calculating recent activity scores")
        
        cutoff_date = datetime.now() - timedelta(days=lookback_months*30)
        recent_scores = []
        
        for tin in competitor_tins:
            query = """
            WITH target_projects AS (
                SELECT
                    b.project_id,
                    p.contract_date
                FROM public_data.thai_project_bid_info b
                JOIN public_data.thai_govt_project p ON b.project_id = p.project_id
                WHERE b.tin = $1 AND p.contract_date >= $3
            ),
            competitor_projects AS (
                SELECT
                    b.project_id,
                    p.contract_date
                FROM public_data.thai_project_bid_info b
                JOIN public_data.thai_govt_project p ON b.project_id = p.project_id
                WHERE b.tin = $2 AND p.contract_date >= $3
            )
            SELECT
                tp.project_id,
                tp.contract_date,
                -- Calculate months ago, ensuring it's not negative
                GREATEST(0, (EXTRACT(EPOCH FROM NOW()) - EXTRACT(EPOCH FROM tp.contract_date))) / (60*60*24*30) AS months_ago
            FROM target_projects tp
            JOIN competitor_projects cp ON tp.project_id = cp.project_id
            WHERE tp.contract_date IS NOT NULL -- Ensure date is not null
            ORDER BY tp.contract_date DESC
            """
            
            result_df = await self.db.read_sql_async(query, [target_tin, tin, cutoff_date])
            
            if result_df.empty:
                recent_scores.append(0.0)
                continue
            
            # Calculate recency score where more recent projects get higher weight
            recency_scores = []
            for _, row in result_df.iterrows():
                months_ago = row['months_ago']
                # Linear decay from 1.0 (now) to 0.0 (lookback_months ago)
                recency_score = max(0, 1 - (months_ago / lookback_months))
                recency_scores.append(recency_score)
            
            # Average of recency scores (0-1 scale)
            avg_score = sum(recency_scores) / len(recency_scores) if recency_scores else 0.0
            recent_scores.append(avg_score)
        
        return pd.Series(recent_scores)
        
    def build_gradient_model(self, data):
        """Build SVM model with the enhanced feature set."""
        logger.info("Building SVM model with enhanced features")
        
        self.data = data
        
        # Select features for the model
        selected_features = [
            'common_projects',           # Number of direct project overlaps
            'common_departments',        # Number of departments in common
            'common_subdepartments',     # Number of subdepartments in common
            'dept_wins',                 # Wins in departments where target company operates
            'total_win_value',           # Total value of won projects
            'win_rate',                  # Percentage of projects won
            'price_range_overlap',       # Project value range similarity
            'recent_activity_score',     # Recent competitive activity
            'dept_engagement_score',     # Department engagement metric
            'subdept_specialization'     # Subdepartment specialization
        ]
        
        # Ensure all selected features exist
        missing_features = [f for f in selected_features if f not in self.data.columns]
        if missing_features:
            logger.error(f"Selected features missing from data: {missing_features}")
            return False
        
        # Prepare features and labels
        X = self.data[selected_features].values
        
        # Create labels for known competitors (1) and others (0)
        y = np.zeros(len(self.data))
        known_indices = self.data[self.data['tin'].isin(self.known_strong_competitors)].index
        y[known_indices] = 1
        
        # Train the model - using SVM instead of gradient descent
        self.model = SVMClassifier(
            C=1.0,                 # Regularization parameter
            kernel='rbf',          # Radial basis function kernel
            gamma='scale',         # Kernel coefficient
            class_weight='balanced' # Handle class imbalance
        )
        self.model.fit(X, y, feature_names=selected_features)
        
        # Calculate model predictions
        y_pred = self.model.predict(X)
        accuracy = np.mean(y_pred == y)
        known_accuracy = np.mean(y_pred[known_indices] == y[known_indices])
        
        logger.info(f"SVM model trained with {len(selected_features)} features")
        logger.info(f"Overall accuracy: {accuracy:.4f}")
        logger.info(f"Accuracy on known competitors: {known_accuracy:.4f}")
        
        # Calculate scores
        self.data['svm_score'] = self.model.predict_proba(X) * 100
        
        return True
    
    def analyze_rankings(self):
        """Analyze the rankings produced by the SVM model."""
        logger.info("Analyzing SVM model rankings")
        
        # Sort by SVM score
        sorted_df = self.data.sort_values('svm_score', ascending=False).reset_index(drop=True)
        sorted_df['rank'] = sorted_df.index + 1
        
        # Find ranks of known competitors
        known_ranks = []
        for tin in self.known_strong_competitors:
            rank_row = sorted_df[sorted_df['tin'] == tin]
            if not rank_row.empty:
                known_ranks.append((
                    rank_row['name'].iloc[0],
                    rank_row['rank'].iloc[0],
                    rank_row['svm_score'].iloc[0]
                ))
        
        # Calculate ranking metrics
        avg_rank = np.mean([r[1] for r in known_ranks])
        median_rank = np.median([r[1] for r in known_ranks])
        top_10_count = sum(1 for r in known_ranks if r[1] <= 10)
        
        # Print results
        print("\n" + "="*80)
        print("SVM MODEL RANKING ANALYSIS")
        print("="*80)
        
        print(f"\nModel features ({len(self.model.feature_names)}):")
        feature_weights = self.model.get_feature_weights()
        for feature, weight in sorted(feature_weights.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feature:<25}: {weight:+.6f}")
        
        print("\nTop 10 competitors by SVM score:")
        top_10 = sorted_df.head(10)[['rank', 'name', 'tin', 'svm_score', 'common_projects', 'win_rate']]
        print(tabulate(top_10, headers='keys', tablefmt='grid', showindex=False))
        
        print("\nKnown strong competitors ranking:")
        for name, rank, score in known_ranks:
            print(f"  {name}: Rank {rank}, Score: {score:.2f}")
        
        print(f"\nRanking metrics:")
        print(f"  Average rank of known competitors: {avg_rank:.2f}")
        print(f"  Median rank of known competitors: {median_rank:.2f}")
        print(f"  Known competitors in top 10: {top_10_count} out of {len(known_ranks)}")
        
        # Save to CSV
        output_csv = os.path.join(self.output_dir, 'svm_scores.csv')
        sorted_df.to_csv(output_csv, index=False)
        logger.info(f"Saved SVM scores to {output_csv}")
        
        return sorted_df
    
    def visualize_results(self):
        """Create visualizations for the gradient model results."""
        logger.info("Creating visualizations...")
        
        # Visualize feature weights
        self._visualize_feature_weights()
        
        # Visualize top companies
        self._visualize_top_companies()
        
        # Visualize feature correlations
        self._visualize_feature_correlations()
    
    def _visualize_feature_weights(self):
        """Visualize feature weights from the model."""
        plt.figure(figsize=(12, 8))
        
        feature_weights = self.model.get_feature_weights()
        features = list(feature_weights.keys())
        weights = list(feature_weights.values())
        
        # Sort by absolute weight
        sorted_indices = np.argsort(np.abs(weights))[::-1]
        features = [features[i] for i in sorted_indices]
        weights = [weights[i] for i in sorted_indices]
        
        # Create colormap based on weight sign
        colors = ['#d73027' if w < 0 else '#4575b4' for w in weights]
        
        bars = plt.barh(features, weights, color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.01 if width > 0 else width - 0.01
            alignment = 'left' if width > 0 else 'right'
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', va='center', ha=alignment, fontweight='bold')
        
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        plt.title('Feature Weights in Enhanced Gradient Model', fontsize=16)
        plt.xlabel('Weight Value', fontsize=12)
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(self.output_dir, 'feature_weights.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature weights visualization saved to {output_file}")
    
    def _visualize_top_companies(self):
        """Visualize top companies by SVM score."""
        # Get top companies
        top_n = 15
        top_companies = self.data.sort_values('svm_score', ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 8))
        
        # Format company names for cleaner display
        display_names = [name[:30] + '...' if len(name) > 30 else name for name in top_companies['name']]
        
        # Create horizontal bar chart
        bars = plt.barh(
            display_names, 
            top_companies['svm_score'],
            color='skyblue',
            edgecolor='gray',
            alpha=0.8
        )
        
        # Highlight known competitors
        for i, (_, row) in enumerate(top_companies.iterrows()):
            if row['tin'] in self.known_strong_competitors:
                bars[i].set_color('red')
                bars[i].set_alpha(1.0)
        
        # Add score labels
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 1, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}', 
                va='center'
            )
        
        plt.title(f'Top {top_n} Companies by SVM Score', fontsize=14)
        plt.xlabel('SVM Score', fontsize=12)
        plt.ylabel('')
        plt.grid(axis='x', alpha=0.3)
        
        # Add legend manually
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Known Strong Competitors')
        blue_patch = mpatches.Patch(color='skyblue', label='Other Companies')
        plt.legend(handles=[red_patch, blue_patch], loc='lower right')
        
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(self.output_dir, 'top_companies.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Top companies visualization saved to {output_file}")

    def _visualize_feature_correlations(self):
        """Visualize correlations between features and SVM score."""
        # Get feature names used in the model
        features = self.model.feature_names
        
        # Add SVM score
        columns_to_correlate = features + ['svm_score']
        
        # Calculate correlation matrix
        corr_matrix = self.data[columns_to_correlate].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            center=0,
            fmt='.2f',
            linewidths=0.5
        )
        plt.title('Feature Correlation Matrix', fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(self.output_dir, 'feature_correlations.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature correlations visualization saved to {output_file}")


async def main():
    """Main entry point for the script."""
    start_time = datetime.now()
    logger.info("Starting Enhanced Gradient Model")
    
    # Initialize database connection
    db = DbConnection()
    await db.initialize()
    
    try:
        # Create model
        model = DatabaseGradientModel(db)
        
        # Get target company information
        target_company = await model.get_target_company_info()
        if not target_company:
            logger.error(f"Target company {model.target_company_tin} not found")
            return
        
        # Get all competitors
        competitors_df = await model.get_all_competitors(lookback_months=60)
        if competitors_df.empty:
            logger.error("No competitors found for analysis")
            return
        
        # Calculate additional metrics
        enhanced_df = await model.calculate_additional_metrics(competitors_df, target_company)
        
        # Build gradient model
        if not model.build_gradient_model(enhanced_df):
            logger.error("Failed to build gradient model")
            return
        
        # Analyze rankings
        sorted_df = model.analyze_rankings()
        
        # Create visualizations
        model.visualize_results()
        
        logger.info("Enhanced gradient model analysis complete")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
    finally:
        # Close database connection
        await db.close()
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Script execution completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    # Ensure event loop compatibility for different OS
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())