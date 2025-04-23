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
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
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


class SklearnPolynomialModel:
    """Gradient Descent Model using scikit-learn's PolynomialFeatures."""
    def __init__(self, degree=2, include_bias=False, interaction_only=False, 
                 regularization=0.01, max_iter=1000):
        """
        Initialize the model with polynomial feature configuration.
        
        Args:
            degree: The degree of the polynomial features (2 = squares and interactions)
            include_bias: Whether to include a bias column (constant feature)
            interaction_only: If True, only include interaction features, no powers
            regularization: L2 regularization strength (C is inverse of regularization)
            max_iter: Maximum number of iterations for logistic regression
        """
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.regularization = regularization
        self.max_iter = max_iter
        
        # Create pipeline with polynomial features, scaling, and logistic regression
        self.pipeline = Pipeline([
            ('polynomial', PolynomialFeatures(degree=degree, 
                                              include_bias=include_bias,
                                              interaction_only=interaction_only)),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(penalty='l2',
                                             C=1/regularization,
                                             max_iter=max_iter,
                                             solver='liblinear'))
        ])
        
        self.feature_names_in_ = None
        self.poly_feature_names_ = None
        
    def fit(self, X, y, feature_names=None):
        """
        Fit the model to the data.
        
        Args:
            X: Feature matrix
            y: Target vector (0 or 1)
            feature_names: Names of the input features
        
        Returns:
            self
        """
        # Store original feature names
        self.feature_names_in_ = (feature_names if feature_names is not None 
                                else [f"feature_{i}" for i in range(X.shape[1])])
        
        # Fit the pipeline
        self.pipeline.fit(X, y)
        
        # Get polynomial feature names
        poly = self.pipeline.named_steps['polynomial']
        self.poly_feature_names_ = poly.get_feature_names_out(self.feature_names_in_)
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.pipeline.predict_proba(X)[:, 1]  # Return probability of class 1
    
    def predict(self, X, threshold=0.5):
        """Predict class labels using a threshold."""
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def get_feature_weights(self):
        """Return dictionary mapping feature names to their weights."""
        classifier = self.pipeline.named_steps['classifier']
        weights = classifier.coef_[0]
        
        if self.poly_feature_names_ is None:
            raise ValueError("Model has not been trained yet")
        
        return dict(zip(self.poly_feature_names_, weights))
    
    def get_feature_importance(self):
        """Return normalized feature importance based on absolute weights."""
        weights_dict = self.get_feature_weights()
        abs_weights = np.abs(list(weights_dict.values()))
        
        # Normalize to sum to 100
        importance = (abs_weights / abs_weights.sum()) * 100
        
        return dict(zip(weights_dict.keys(), importance))
    
    @property
    def feature_names(self):
        """Get the input feature names (for compatibility with the previous model)."""
        return self.feature_names_in_
    
    def evaluate_cv(self, X, y, cv=5):
        """Evaluate model with cross-validation."""
        # Accuracy
        accuracy = cross_val_score(self.pipeline, X, y, cv=cv, scoring='accuracy')
        
        # AUC
        auc = cross_val_score(self.pipeline, X, y, cv=cv, scoring='roc_auc')
        
        return {
            'accuracy_mean': accuracy.mean(),
            'accuracy_std': accuracy.std(),
            'auc_mean': auc.mean(),
            'auc_std': auc.std()
        }


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
        self.output_dir = 'poly_output'
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
        """Build gradient descent model with sklearn polynomial features."""
        logger.info("Building gradient model with sklearn polynomial features")
        
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
        
        # Create and train the model
        self.model = SklearnPolynomialModel(
            degree=2,                 # Square and interaction terms
            include_bias=False,       # No bias term (handled separately)
            interaction_only=False,   # Include both interactions and powers
            regularization=0.1,       # L2 regularization strength
            max_iter=2000             # Maximum iterations for convergence
        )
        
        self.model.fit(X, y, feature_names=selected_features)
        
        # Calculate model predictions
        y_pred = self.model.predict(X)
        accuracy = np.mean(y_pred == y)
        known_accuracy = np.mean(y_pred[known_indices] == y[known_indices])
        
        # Get count of features
        total_features = len(self.model.poly_feature_names_)
        original_features = len(selected_features)
        added_features = total_features - original_features
        
        logger.info(f"Model trained with {original_features} original features and {added_features} polynomial features")
        logger.info(f"Total features: {total_features}")
        logger.info(f"Overall accuracy: {accuracy:.4f}")
        logger.info(f"Accuracy on known competitors: {known_accuracy:.4f}")
        
        # Get performance metrics with cross-validation
        cv_metrics = self.model.evaluate_cv(X, y, cv=5)
        logger.info(f"Cross-validation AUC: {cv_metrics['auc_mean']:.4f} ± {cv_metrics['auc_std']:.4f}")
        
        # Calculate gradient scores
        self.data['gradient_score'] = self.model.predict_proba(X) * 100
        
        return True

    def evaluate_models_with_cv(self, data, feature_sets=None, k_folds=5):
        """
        Evaluate different models with sklearn's polynomial features using k-fold cross-validation.
        
        Args:
            data: DataFrame with features and target labels
            feature_sets: Dict of feature set name to list of feature names
            k_folds: Number of folds for cross-validation
        
        Returns:
            DataFrame with evaluation results
        """
        logger.info(f"Evaluating models with {k_folds}-fold cross-validation using sklearn")
        
        if feature_sets is None:
            # Default feature sets to compare
            selected_features = [
                'common_projects',
                'common_departments',
                'common_subdepartments',
                'dept_wins',
                'total_win_value',
                'win_rate',
                'price_range_overlap',
                'recent_activity_score',
                'dept_engagement_score',
                'subdept_specialization'
            ]
            
            feature_sets = {
                "Basic": ['common_projects', 'common_departments', 'win_rate'],
                "Standard": ['common_projects', 'common_departments', 'dept_wins', 
                        'total_win_value', 'win_rate'],
                "Full": selected_features,
            }
        
        # Configurations to test
        model_configs = [
            {
                "name": "Linear",
                "degree": 1,  # Linear model, no polynomial features
                "interaction_only": False,
                "regularization": 0.1
            },
            {
                "name": "Poly (Degree 2)",
                "degree": 2,  # Square terms and interactions
                "interaction_only": False,
                "regularization": 0.1
            },
            {
                "name": "Interactions Only",
                "degree": 2,
                "interaction_only": True,  # Only interaction terms, no squared terms
                "regularization": 0.1
            }
        ]
        
        # Prepare results storage
        results = []
        
        # Create labels for known competitors (1) and others (0)
        known_competitors = np.zeros(len(data))
        known_indices = data[data['tin'].isin(self.known_strong_competitors)].index
        known_competitors[known_indices] = 1
        
        # Run cross-validation for each feature set and model config
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        for feature_set_name, features in feature_sets.items():
            # Ensure all features exist
            missing_features = [f for f in features if f not in data.columns]
            if missing_features:
                logger.warning(f"Feature set '{feature_set_name}' has missing features: {missing_features}")
                continue
            
            # Extract feature matrix
            X = data[features].values
            y = known_competitors
            
            for config in model_configs:
                model_name = config["name"]
                logger.info(f"Evaluating {model_name} model with {feature_set_name} features")
                
                model = SklearnPolynomialModel(
                    degree=config["degree"],
                    interaction_only=config["interaction_only"],
                    regularization=config["regularization"]
                )
                
                metrics = model.evaluate_cv(X, y, cv=kf)
                
                # Store results
                results.append({
                    "Feature Set": feature_set_name,
                    "Model": model_name,
                    "Accuracy": metrics["accuracy_mean"],
                    "Accuracy Std": metrics["accuracy_std"],
                    "AUC": metrics["auc_mean"],
                    "AUC Std": metrics["auc_std"],
                    "Feature Count": len(features),
                    "Degree": config["degree"],
                    "Interactions Only": config["interaction_only"]
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Print results
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS (CROSS-VALIDATION)")
        print("="*80 + "\n")
        print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))
        
        # Save to CSV
        output_csv = os.path.join(self.output_dir, 'sklearn_model_evaluation.csv')
        results_df.to_csv(output_csv, index=False)
        logger.info(f"Saved model evaluation results to {output_csv}")
        
        # Visualize results
        self._visualize_model_comparison(results_df)
        
        return results_df
    
    def _visualize_model_comparison(self, results_df):
        """Visualize model comparison results."""
        plt.figure(figsize=(12, 8))
        
        # Group by feature set and model
        feature_sets = results_df['Feature Set'].unique()
        models = results_df['Model'].unique()
        
        # Set up the plot
        x = np.arange(len(feature_sets))
        width = 0.25  # Width of the bars
        
        # Create grouped bar chart for AUC scores
        for i, model in enumerate(models):
            model_results = results_df[results_df['Model'] == model]
            auc_values = []
            errors = []
            
            for feature_set in feature_sets:
                feature_set_results = model_results[model_results['Feature Set'] == feature_set]
                if not feature_set_results.empty:
                    auc_values.append(feature_set_results['AUC'].values[0])
                    errors.append(feature_set_results['AUC Std'].values[0])
                else:
                    auc_values.append(0)
                    errors.append(0)
            
            # Position bars with offset
            offset = (i - len(models)/2 + 0.5) * width
            bars = plt.bar(x + offset, auc_values, width, label=model)
            
            # Add error bars
            plt.errorbar(x + offset, auc_values, yerr=errors, fmt='none', capsize=5, ecolor='black', alpha=0.6)
            
            # Add value labels
            for j, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Customize plot
        plt.ylabel('AUC Score')
        plt.title('Model Performance Comparison (AUC)')
        plt.xticks(x, feature_sets)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Models')
        
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        logger.info(f"Model comparison visualization saved to {output_file}")
    
    def analyze_rankings(self):
        """Analyze the rankings produced by the gradient model."""
        logger.info("Analyzing gradient model rankings")
        
        # Sort by gradient score
        sorted_df = self.data.sort_values('gradient_score', ascending=False).reset_index(drop=True)
        sorted_df['rank'] = sorted_df.index + 1
        
        # Find ranks of known competitors
        known_ranks = []
        for tin in self.known_strong_competitors:
            rank_row = sorted_df[sorted_df['tin'] == tin]
            if not rank_row.empty:
                known_ranks.append((
                    rank_row['name'].iloc[0],
                    rank_row['rank'].iloc[0],
                    rank_row['gradient_score'].iloc[0]
                ))
        
        # Calculate ranking metrics
        avg_rank = np.mean([r[1] for r in known_ranks]) if known_ranks else np.nan
        median_rank = np.median([r[1] for r in known_ranks]) if known_ranks else np.nan
        top_10_count = sum(1 for r in known_ranks if r[1] <= 10)
        
        # Print results
        print("\n" + "="*80)
        print("ENHANCED GRADIENT MODEL RANKING ANALYSIS")
        print("="*80)
        
        print(f"\nModel features ({len(self.model.poly_feature_names_)}):")
        feature_weights = self.model.get_feature_weights()
        for feature, weight in sorted(feature_weights.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feature:<25}: {weight:+.6f}")
        
        print("\nTop 10 competitors by gradient score:")
        top_10 = sorted_df.head(10)[['rank', 'name', 'tin', 'gradient_score', 'common_projects', 'win_rate']]
        print(tabulate(top_10, headers='keys', tablefmt='grid', showindex=False))
        
        print("\nKnown strong competitors ranking:")
        for name, rank, score in known_ranks:
            print(f"  {name}: Rank {rank}, Score: {score:.2f}")
        
        print(f"\nRanking metrics:")
        print(f"  Average rank of known competitors: {avg_rank:.2f}")
        print(f"  Median rank of known competitors: {median_rank:.2f}")
        print(f"  Known competitors in top 10: {top_10_count} out of {len(known_ranks)}")
        
        # Save to CSV
        output_csv = os.path.join(self.output_dir, 'gradient_scores.csv')
        sorted_df.to_csv(output_csv, index=False)
        logger.info(f"Saved gradient scores to {output_csv}")
        
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
        plt.figure(figsize=(14, 10))
        
        feature_weights = self.model.get_feature_weights()
        features = list(feature_weights.keys())
        weights = list(feature_weights.values())
        
        # Categorize features by their type (original, squared, interaction, etc.)
        feature_types = {}
        for feature in features:
            if feature.endswith("^2"):
                category = "Squared"
            elif " " in feature:
                category = "Interaction"
            else:
                category = "Original"
            
            if category not in feature_types:
                feature_types[category] = []
            
            feature_types[category].append(feature)
        
        # Sort each category by absolute weight
        for category, feat_list in feature_types.items():
            feature_types[category] = sorted(
                feat_list, 
                key=lambda f: abs(feature_weights[f]),
                reverse=True
            )
        
        # Define colors for each category
        category_colors = {
            "Original": "#4575b4",  # Blue
            "Squared": "#d73027",   # Red
            "Interaction": "#91cf60" # Green
        }
        
        # Prepare data for grouped bar chart
        all_features = []
        all_weights = []
        all_colors = []
        
        # Use a fixed order for categories
        category_order = ["Original", "Squared", "Interaction"]
        for category in category_order:
            if category in feature_types:
                for feature in feature_types[category]:
                    all_features.append(feature)
                    all_weights.append(feature_weights[feature])
                    all_colors.append(category_colors[category])
        
        # Sort by absolute weight within each category
        sorted_indices = np.argsort(np.abs(all_weights))[::-1]
        all_features = [all_features[i] for i in sorted_indices]
        all_weights = [all_weights[i] for i in sorted_indices]
        all_colors = [all_colors[i] for i in sorted_indices]
        
        # Truncate to top N features if there are too many
        max_features = 30
        if len(all_features) > max_features:
            all_features = all_features[:max_features]
            all_weights = all_weights[:max_features]
            all_colors = all_colors[:max_features]
        
        # Create horizontal bar chart
        bars = plt.barh(all_features, all_weights, color=all_colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.01 if width > 0 else width - 0.01
            alignment = 'left' if width > 0 else 'right'
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', va='center', ha=alignment, fontweight='bold')
        
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        plt.title('Feature Weights in Enhanced Polynomial Model', fontsize=16)
        plt.xlabel('Weight Value', fontsize=12)
        
        # Add legend for feature types
        import matplotlib.patches as mpatches
        legend_patches = [mpatches.Patch(color=category_colors[cat], label=cat) 
                         for cat in category_order if cat in feature_types]
        plt.legend(handles=legend_patches, loc='lower right')
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(self.output_dir, 'polynomial_feature_weights.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Polynomial feature weights visualization saved to {output_file}")
    
    def _visualize_top_companies(self):
        """Visualize top companies by gradient score."""
        # Get top companies
        top_n = 15
        top_companies = self.data.sort_values('gradient_score', ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 8))
        
        # Format company names for cleaner display
        display_names = [name[:30] + '...' if len(name) > 30 else name for name in top_companies['name']]
        
        # Create horizontal bar chart
        bars = plt.barh(
            display_names, 
            top_companies['gradient_score'],
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
        
        plt.title(f'Top {top_n} Companies by Polynomial Model Score', fontsize=14)
        plt.xlabel('Score', fontsize=12)
        plt.ylabel('')
        plt.grid(axis='x', alpha=0.3)
        
        # Add legend
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
        """Visualize correlations between original features and gradient score."""
        # Use only the original features, not polynomial ones
        original_features = self.model.feature_names_in_
        
        # Add gradient score
        columns_to_correlate = list(original_features) + ['gradient_score']
        
        # Calculate correlation matrix
        corr_matrix = self.data[columns_to_correlate].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            center=0,
            fmt='.2f',
            linewidths=0.5,
            mask=mask
        )
        plt.title('Feature Correlation Matrix', fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(self.output_dir, 'feature_correlations.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature correlations visualization saved to {output_file}")


async def main():
    """Main entry point for the script with polynomial feature support."""
    start_time = datetime.now()
    logger.info("Starting Enhanced Gradient Model with Polynomial Features")
    
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
        
        # Run model evaluation with cross-validation
        logger.info("Running model evaluation with cross-validation")
        feature_sets = {
            "Basic": ['common_projects', 'common_departments', 'win_rate'],
            "Standard": ['common_projects', 'common_departments', 'dept_wins', 
                       'total_win_value', 'win_rate'],
            "Full": ['common_projects', 'common_departments', 'common_subdepartments',
                   'dept_wins', 'total_win_value', 'win_rate', 'price_range_overlap',
                   'recent_activity_score', 'dept_engagement_score', 'subdept_specialization']
        }
        evaluation_results = model.evaluate_models_with_cv(enhanced_df, feature_sets)
        
        # Build the best gradient model based on evaluation
        # For this example, we'll use the full model with square and sqrt features
        logger.info("Building the best polynomial gradient model")
        if not model.build_gradient_model(enhanced_df):
            logger.error("Failed to build gradient model")
            return
        
        # Analyze rankings
        sorted_df = model.analyze_rankings()
        
        # Create visualizations
        model.visualize_results()
        
        logger.info("Enhanced gradient model analysis with polynomial features complete")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Close database connection
        await db.close()
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Script execution completed in {elapsed_time:.2f} seconds")


# This is crucial - actually run the main function when the script is executed
if __name__ == "__main__":
    # Ensure event loop compatibility for different OS
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())