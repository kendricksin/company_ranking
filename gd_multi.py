#!/usr/bin/env python3
"""
Enhanced Multi-Reference Gradient Model for Competitor Scoring with Comprehensive Validation

This script implements an improved gradient model that:
1. Learns from multiple reference companies
2. Implements cross-validation to test generalization
3. Tests stability across random seeds
4. Uses bootstrap analysis for feature importance confidence intervals
5. Improves learning with scheduling and regularization
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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from tabulate import tabulate
import logging
import asyncpg
from dotenv import load_dotenv
from datetime import datetime, timedelta
import random
from tqdm import tqdm  # For progress bars, install with pip if needed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Import needed connection class from original script
try:
    from gd_new_score import DbConnection
except ImportError:
    # Define DbConnection here if the import fails
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


class EnhancedGradientDescentClassifier:
    """Enhanced Logistic Regression Classifier with improved stability and learning rate scheduling."""
    def __init__(self, learning_rate=0.01, iterations=1000, regularization=0.01, random_seed=42, use_lr_scheduling=True):
        self.lr = learning_rate
        self.initial_lr = learning_rate
        self.iter = iterations
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.random_seed = random_seed
        self.use_lr_scheduling = use_lr_scheduling
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def sigmoid(self, z):
        # Clip input to avoid overflow in exp
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def fit(self, X, y, feature_names=None):
        n_samples, n_features = X.shape
        self.feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(n_features)]

        # Reset random seed again before training
        np.random.seed(self.random_seed)
        
        # Feature Scaling
        X_scaled = self.scaler.fit_transform(X)

        # Initialize weights with small random values instead of zeros
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0

        # Gradient Descent with learning rate scheduling
        for i in range(self.iter):
            # Update learning rate if scheduling is enabled
            if self.use_lr_scheduling:
                self.lr = self.initial_lr / (1 + 0.01 * i)  # Gradually decrease learning rate
            
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


class MultiReferenceGradientModel:
    """Model that can learn from multiple reference companies and their competitors."""
    
    def __init__(self, db_conn, random_seed=42):
        """Initialize with database connection."""
        self.db = db_conn
        self.data = None
        self.model = None
        self.feature_names = None
        self.debug = True  # Set to True for detailed debug information
        self.random_seed = random_seed
        
        # Set random seeds
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # Define reference companies and their known competitors
        self.reference_companies = [
            {
                "name": "Italian-Thai",
                "target": "0107537000939",  # Italian Thai
                "competitors": [
                    "0107536001001",  # ซิโน-ไทย
                    "0105509002602",  # ซีวิล คอนสตรัคชั่น
                    "0107545000217",  # เพาเวอร์ไลน์
                ]
            },
            {
                "name": "Ruangritthai",
                "target": "0105543041542",  # บริษัท เรืองฤทัย จำกัด
                "competitors": [
                    "0105540085581",  # บริษัท ตรีสกุล จำกัด
                    "0103503000975",  # แสงนิยม
                    "0105538044211",  # ปาล์ม คอน
                ]
            },
            {
                "name": "PYS",
                "target": "0105561013814",  # บริษัท พี.วาย.เอส.คอนสตรั๊คชั่น จำกัด
                "competitors": [
                    "0105538044211",  # บริษัท ปาล์ม คอน จำกัด
                    "0105539121707",  # บริษัท ที.พี.ซี. คอนกรีตอัดแรง จำกัด
                    "0105553059231",  # บริษัท กรุงไทยสถาปัตย์ จำกัด
                ]
            }
        ]

        # Set output directory
        self.output_dir = 'results/enhanced_model_output'
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def get_company_info(self, target_tin):
        """Get detailed information about a company."""
        logger.info(f"Fetching company info for target: {target_tin}")
        
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
        
        company_df = await self.db.read_sql_async(query, [target_tin])
        
        if company_df.empty:
            logger.warning(f"No company information found for target: {target_tin}")
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
        
        logger.info(f"Company info: {company_data['name']} ({target_tin})")
        return company_data
    
    async def get_competitors(self, target_tin, lookback_months=60):
        """Get all competitors for a target company from the database."""
        logger.info(f"Fetching competitors for company TIN: {target_tin}")
        
        time_limit_condition = ""
        params = [target_tin]
        
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
            logger.warning(f"No competitors found for company TIN: {target_tin} within the lookback period.")
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
            result_df['tin'].tolist(), target_company['tin']
        )
        result_df['price_range_overlap'] = price_range_overlap
        
        # Calculate recent activity score
        recent_activity = await self._calculate_recent_activity(
            result_df['tin'].tolist(), target_company['tin']
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
                if null_count > 0 and self.debug:
                    logger.warning(f"Column '{col}' has {null_count} null values, filling with 0")
                result_df[col] = result_df[col].fillna(0)
        
        return result_df
    
    async def _calculate_price_range_overlap(self, competitor_tins, target_tin):
        """Calculate price range overlap between target and competitors."""
        if self.debug:
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
        if self.debug:
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
    
    async def build_combined_dataset(self):
        """Build a combined dataset from all reference companies."""
        logger.info("Building combined dataset from all reference companies")
        combined_data = []
        
        for reference in self.reference_companies:
            logger.info(f"Processing reference company: {reference['name']} ({reference['target']})")
            
            # Get target company information
            target_company = await self.get_company_info(reference["target"])
            if not target_company:
                logger.warning(f"Target company {reference['target']} not found, skipping")
                continue
                
            # Get all competitors
            competitors_df = await self.get_competitors(reference["target"], lookback_months=60)
            if competitors_df.empty:
                logger.warning(f"No competitors found for {reference['target']}, skipping")
                continue
            
            # Log the target company info and sample competitors
            logger.info(f"Target company info: {target_company['name']} ({reference['target']})")
            logger.info(f"Found {len(competitors_df)} competitors for this target")

            # Check if known competitors are in the competitors_df
            for tin in reference["competitors"]:
                comp_match = competitors_df[competitors_df['tin'] == tin]
                if comp_match.empty:
                    logger.warning(f"Known competitor {tin} not found in competitors data")
                else:
                    logger.info(f"Known competitor found: {comp_match['name'].values[0]} ({tin})")
                
            # Calculate additional metrics
            enhanced_df = await self.calculate_additional_metrics(competitors_df, target_company)
            
            # Mark which competitors are known for this target
            enhanced_df['is_known_competitor'] = enhanced_df['tin'].isin(reference["competitors"]).astype(int)
            enhanced_df['source_target'] = reference["target"]
            enhanced_df['source_name'] = reference["name"]
            
            # Add to combined data
            combined_data.append(enhanced_df)
            logger.info(f"Added {len(enhanced_df)} competitors for {reference['name']}")
        
        if not combined_data:
            logger.error("No data collected from any reference company")
            return None
            
        # Combine all datasets
        combined_df = pd.concat(combined_data, ignore_index=True)
        logger.info(f"Combined dataset created with {len(combined_df)} entries from {len(self.reference_companies)} reference points")
        
        return combined_df
    
    async def train_unified_model(self, use_polynomial_features=False, degree=2):
        """Train a unified model using data from all reference companies."""
        logger.info("Training unified model")
        
        # Build combined dataset
        combined_df = await self.build_combined_dataset()
        if combined_df is None:
            logger.error("Failed to build combined dataset")
            return False
        
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
        missing_features = [f for f in selected_features if f not in combined_df.columns]
        if missing_features:
            logger.error(f"Selected features missing from data: {missing_features}")
            return False
        
        # Store original feature names
        self.base_features = selected_features
        self.use_polynomial_features = use_polynomial_features
        self.poly_degree = degree
        self.poly_transformer = None
        
        # Prepare features and labels
        X = combined_df[selected_features].values
        y = combined_df['is_known_competitor'].values
        
        # Apply polynomial features if requested
        if use_polynomial_features:
            logger.info(f"Applying polynomial features (degree={degree})")
            self.poly_transformer = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
            X_poly = self.poly_transformer.fit_transform(X)
            poly_feature_names = self.poly_transformer.get_feature_names_out(self.base_features)
            
            # Use polynomial features
            X = X_poly
            self.feature_names = poly_feature_names
            
            logger.info(f"Expanded to {len(poly_feature_names)} polynomial features")
        else:
            # Use original features
            self.feature_names = selected_features
        
        # Train the model
        self.model = EnhancedGradientDescentClassifier(
            learning_rate=0.05, 
            iterations=2500,  
            regularization=0.03,
            random_seed=self.random_seed,
            use_lr_scheduling=True
        )
        self.model.fit(X, y, feature_names=self.feature_names)
        
        # Store data for analysis
        self.data = combined_df
        
        # Calculate model predictions and scores
        y_pred = self.model.predict(X)
        combined_df['gradient_score'] = self.model.predict_proba(X) * 100
        
        # Overall accuracy
        accuracy = np.mean(y_pred == y)
        logger.info(f"Overall accuracy: {accuracy:.4f}")
        
        # Per-reference company accuracy
        self._evaluate_per_company_performance()
        
        return True
    
    def _evaluate_per_company_performance(self):
        """Evaluate how well the unified model performs for each reference company."""
        if self.data is None or self.model is None:
            return
        
        logger.info("Per-reference company performance evaluation:")
        
        for reference in self.reference_companies:
            target = reference["target"]
            company_name = reference["name"]
            
            # Filter data for this target
            target_data = self.data[self.data['source_target'] == target].copy()
            
            if target_data.empty:
                logger.warning(f"No data available for {company_name}")
                continue
            
            # Get features for this target
            if self.use_polynomial_features and self.poly_transformer is not None:
                # Need to apply polynomial transformation to the base features first
                X_base = target_data[self.base_features].values
                X = self.poly_transformer.transform(X_base)
            else:
                # Use original features directly
                X = target_data[self.base_features].values
                
            # Calculate accuracy for this target
            y_true = target_data['is_known_competitor'].values
            y_pred = self.model.predict(X)
            accuracy = np.mean(y_pred == y_true)
            
            # Get known competitor ranks
            target_data = target_data.sort_values('gradient_score', ascending=False).reset_index(drop=True)
            target_data['rank'] = target_data.index + 1
            
            # Find ranks of known competitors
            known_competitor_ranks = []
            for tin in reference["competitors"]:
                rank_row = target_data[target_data['tin'] == tin]
                if not rank_row.empty:
                    rank = rank_row['rank'].iloc[0]
                    known_competitor_ranks.append(rank)
            
            if known_competitor_ranks:
                avg_rank = np.mean(known_competitor_ranks)
                top_10_pct = sum(1 for r in known_competitor_ranks if r <= 10) / len(known_competitor_ranks) * 100
                
                logger.info(f"  {company_name} ({target}):")
                logger.info(f"    Accuracy: {accuracy:.4f}")
                logger.info(f"    Known competitor ranks: {known_competitor_ranks}")
                logger.info(f"    Average rank: {avg_rank:.2f}")
                logger.info(f"    Percent in top 10: {top_10_pct:.1f}%")
    
    def analyze_feature_weights(self):
        """Analyze the feature weights from the unified model."""
        if self.model is None:
            logger.error("Model not trained yet")
            return
        
        logger.info("Analyzing feature weights from unified model")
        
        # Get feature weights
        feature_weights = self.model.get_feature_weights()
        
        # Print feature weights
        print("\n" + "="*80)
        print("UNIFIED GRADIENT MODEL FEATURE WEIGHTS")
        print("="*80)
        
        print("\nFeature weights (sorted by absolute value):")
        for feature, weight in sorted(feature_weights.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feature:<45}: {weight:+.6f}")
        
        # Visualize feature weights (limit to top 15 if there are many)
        self._visualize_feature_weights(feature_weights)
        
        return feature_weights
    
    def _visualize_feature_weights(self, feature_weights):
        """Visualize feature weights from the model."""
        plt.figure(figsize=(12, 10))
        
        features = list(feature_weights.keys())
        weights = list(feature_weights.values())
        
        # Sort by absolute weight
        sorted_indices = np.argsort(np.abs(weights))[::-1]
        
        # Limit to top 20 features if there are many
        if len(features) > 20:
            sorted_indices = sorted_indices[:20]
            
        features = [features[i] for i in sorted_indices]
        weights = [weights[i] for i in sorted_indices]
        
        # Format feature names for display (shorten if needed)
        display_features = features.copy()
        if any('_' in f for f in features):
            # For polynomial features, make them more readable
            display_features = [f.replace(' ', '').replace('_', ' × ') if ' ' in f else f for f in display_features]
            # Truncate if too long
            display_features = [f[:40] + '...' if len(f) > 40 else f for f in display_features]
        
        # Create colormap based on weight sign
        colors = ['#d73027' if w < 0 else '#4575b4' for w in weights]
        
        bars = plt.barh(display_features, weights, color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.01 if width > 0 else width - 0.01
            alignment = 'left' if width > 0 else 'right'
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', va='center', ha=alignment, fontweight='bold')
        
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        plt.title('Top Feature Weights in Unified Gradient Model', fontsize=16)
        plt.xlabel('Weight Value', fontsize=12)
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(self.output_dir, 'unified_feature_weights.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Unified feature weights visualization saved to {output_file}")
    
    async def apply_to_new_target(self, target_tin, target_name=None):
        """Apply the unified model to a new target company."""
        if self.model is None:
            logger.error("Model not trained yet")
            return None
        
        logger.info(f"Applying unified model to new target: {target_tin}")
        
        # Get target company information
        target_company = await self.get_company_info(target_tin)
        if not target_company:
            logger.error(f"Target company {target_tin} not found")
            return None
        
        # Use the target name from the data if not provided
        if target_name is None:
            target_name = target_company.get('name', 'Unknown')
        
        logger.info(f"Processing target: {target_name}")
        
        # Get competitors
        competitors_df = await self.get_competitors(target_tin, lookback_months=60)
        if competitors_df.empty:
            logger.error(f"No competitors found for {target_tin}")
            return None
        
        # Calculate additional metrics
        enhanced_df = await self.calculate_additional_metrics(competitors_df, target_company)
        
        # Apply the unified model
        if self.use_polynomial_features and self.poly_transformer is not None:
            # Use stored polynomial transformer
            X_base = enhanced_df[self.base_features].values
            X = self.poly_transformer.transform(X_base)
        else:
            # Using original feature set
            X = enhanced_df[self.base_features].values
        
        enhanced_df['gradient_score'] = self.model.predict_proba(X) * 100
        
        # Sort and rank competitors
        sorted_df = enhanced_df.sort_values('gradient_score', ascending=False).reset_index(drop=True)
        sorted_df['rank'] = sorted_df.index + 1
        
        logger.info(f"Applied unified model to {target_name}, found {len(sorted_df)} potential competitors")
        
        # Print top competitors
        print("\n" + "="*80)
        print(f"TOP COMPETITORS FOR {target_name.upper()} USING UNIFIED MODEL")
        print("="*80)
        
        print("\nTop 10 competitors by gradient score:")
        top_10 = sorted_df.head(10)[['rank', 'name', 'tin', 'gradient_score', 'common_projects', 'win_rate']]
        print(tabulate(top_10, headers='keys', tablefmt='grid', showindex=False))
        
        # Save to CSV
        output_csv = os.path.join(self.output_dir, f'unified_scores_{target_tin}.csv')
        sorted_df.to_csv(output_csv, index=False)
        logger.info(f"Saved gradient scores to {output_csv}")
        
        # Visualize top companies
        self._visualize_top_companies(sorted_df, target_name)
        
        return sorted_df
    
    def _visualize_top_companies(self, sorted_df, target_name):
        """Visualize top companies for a target."""
        # Get top companies
        top_n = 15
        top_companies = sorted_df.head(top_n)
        
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
        
        # Add score labels
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 1, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}', 
                va='center'
            )
        
        plt.title(f'Top {top_n} Competitors for {target_name}', fontsize=14)
        plt.xlabel('Gradient Score', fontsize=12)
        plt.ylabel('')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(self.output_dir, f'top_companies_{target_name.replace(" ", "_")}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Top companies visualization saved to {output_file}")


class CrossCompanyValidator:
    """Handles leave-one-out cross-validation across companies."""
    
    def __init__(self, multi_model, db_conn):
        self.multi_model = multi_model
        self.db = db_conn
        self.results = []
        self.output_dir = 'results/cross_validation_output'
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def validate(self):
        """Perform leave-one-out cross-validation."""
        companies = self.multi_model.reference_companies
        validation_results = []
        
        print("\n" + "="*80)
        print("CROSS-COMPANY VALIDATION")
        print("="*80)
        
        # Get polynomial features settings from the source model
        use_poly = self.multi_model.use_polynomial_features if hasattr(self.multi_model, 'use_polynomial_features') else False
        poly_degree = self.multi_model.poly_degree if hasattr(self.multi_model, 'poly_degree') else 2
        
        # For each possible train-test split
        for test_idx in range(len(companies)):
            # Use one company as test set
            test_company = companies[test_idx]
            # Use others as training set
            train_companies = [c for i, c in enumerate(companies) if i != test_idx]
            
            print(f"\n==== Cross-validation fold {test_idx+1} ====")
            print(f"Test company: {test_company['name']}")
            print(f"Train companies: {[c['name'] for c in train_companies]}")
            
            # Train on subset of companies
            temp_model = MultiReferenceGradientModel(self.db, random_seed=42)
            temp_model.reference_companies = train_companies
            
            # Train model with consistent polynomial settings
            success = await temp_model.train_unified_model(use_polynomial_features=use_poly, degree=poly_degree)
            if not success:
                print(f"Failed to train model for fold {test_idx+1}")
                continue
                
            # Apply to test company
            results = await temp_model.apply_to_new_target(
                test_company['target'], 
                test_company['name']
            )
            
            if results is None:
                print(f"Failed to apply model to test company in fold {test_idx+1}")
                continue
            
            # Evaluate how well it ranks known competitors
            known_ranks = []
            for tin in test_company['competitors']:
                comp_row = results[results['tin'] == tin]
                if not comp_row.empty:
                    rank = comp_row['rank'].iloc[0]
                    score = comp_row['gradient_score'].iloc[0]
                    name = comp_row['name'].iloc[0]
                    known_ranks.append((tin, name, rank, score))
            
            # Save metrics
            fold_results = {
                'fold': test_idx + 1,
                'test_company': test_company['name'],
                'known_competitor_ranks': [r[2] for r in known_ranks],
                'known_competitor_names': [r[1] for r in known_ranks],
                'avg_rank': np.mean([r[2] for r in known_ranks]) if known_ranks else float('inf'),
                'top_10_pct': sum(1 for r in known_ranks if r[2] <= 10)/len(known_ranks)*100 if known_ranks else 0,
                'feature_weights': temp_model.model.get_feature_weights()
            }
            validation_results.append(fold_results)
            
            print(f"Known competitors:")
            for tin, name, rank, score in known_ranks:
                print(f"  {name} (rank: {rank}, score: {score:.2f})")
            
            if known_ranks:
                print(f"Average rank: {fold_results['avg_rank']:.2f}")
                print(f"Percent in top 10: {fold_results['top_10_pct']:.1f}%")
        
        # Save results
        self.results = validation_results
        
        # Print summary
        self._print_summary()
        
        # Visualize results
        self._visualize_cross_validation_results()
        
        return validation_results
    
    def _print_summary(self):
        """Print a summary of cross-validation results."""
        if not self.results:
            return
            
        print("\n" + "="*80)
        print("CROSS-VALIDATION SUMMARY")
        print("="*80)
        
        # Print summary table
        summary_data = []
        for result in self.results:
            summary_data.append([
                result['fold'],
                result['test_company'],
                f"{', '.join(map(str, result['known_competitor_ranks']))}",
                f"{result['avg_rank']:.2f}",
                f"{result['top_10_pct']:.1f}%"
            ])
        
        print("\nPerformance across folds:")
        print(tabulate(
            summary_data,
            headers=['Fold', 'Test Company', 'Known Competitor Ranks', 'Avg Rank', 'Top 10%'],
            tablefmt='grid'
        ))
        
        # Calculate overall metrics
        avg_ranks = [r['avg_rank'] for r in self.results]
        top10_pcts = [r['top_10_pct'] for r in self.results]
        
        print(f"\nOverall cross-validation performance:")
        print(f"  Average rank of known competitors: {np.mean(avg_ranks):.2f}")
        print(f"  Average % of known competitors in top 10: {np.mean(top10_pcts):.1f}%")
    
    def _visualize_cross_validation_results(self):
        """Visualize the cross-validation results."""
        if not self.results:
            return
            
        # 1. Visualize ranks across folds
        plt.figure(figsize=(10, 6))
        
        fold_labels = [f"Fold {r['fold']}\n({r['test_company']})" for r in self.results]
        avg_ranks = [r['avg_rank'] for r in self.results]
        
        bars = plt.bar(fold_labels, avg_ranks, color='skyblue')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.1,
                f'{height:.1f}',
                ha='center', va='bottom'
            )
        
        plt.title('Average Rank of Known Competitors Across Cross-Validation Folds', fontsize=14)
        plt.ylabel('Average Rank', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(self.output_dir, 'cross_val_avg_ranks.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Visualize feature weight consistency across folds
        # For simplicity, focus on the most important features
        all_weights = {}
        for result in self.results:
            weights = result['feature_weights']
            for feature, weight in weights.items():
                if feature not in all_weights:
                    all_weights[feature] = []
                all_weights[feature].append(weight)
        
        # Get features that appear in all folds
        common_features = [f for f, w in all_weights.items() if len(w) == len(self.results)]
        
        # Sort by average absolute weight
        sorted_features = sorted(
            common_features,
            key=lambda f: np.mean(np.abs(all_weights[f])),
            reverse=True
        )
        
        # Take top 10 features
        top_features = sorted_features[:min(10, len(sorted_features))]
        
        # Create box plot of weights
        plt.figure(figsize=(12, 8))
        
        # Prepare data for boxplot
        boxplot_data = [all_weights[f] for f in top_features]
        
        # Format feature names for display
        display_features = top_features.copy()
        if any('_' in f for f in top_features):
            # For polynomial features, make them more readable
            display_features = [f.replace(' ', '').replace('_', ' × ') for f in display_features]
            # Truncate if too long
            display_features = [f[:40] + '...' if len(f) > 40 else f for f in display_features]
        
        # Create boxplot
        plt.boxplot(
            boxplot_data,
            vert=False,
            labels=display_features,
            patch_artist=True,
            boxprops=dict(facecolor='skyblue', alpha=0.8)
        )
        
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        plt.title('Feature Weight Consistency Across Cross-Validation Folds', fontsize=14)
        plt.xlabel('Weight Value', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(self.output_dir, 'cross_val_weight_consistency.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()


class RandomSeedValidator:
    """Validates model stability across different random seeds."""
    
    def __init__(self, multi_model, db_conn):
        self.multi_model = multi_model
        self.db = db_conn
        self.weight_history = {}
        self.rank_stability = {}
        self.output_dir = 'results/seed_validation_output'
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def validate(self, num_seeds=10):
        """Perform validation with multiple random seeds."""
        print("\n" + "="*80)
        print("RANDOM SEED STABILITY ANALYSIS")
        print("="*80)
        
        # Use base features from the multi-reference model if available
        if hasattr(self.multi_model, 'base_features') and self.multi_model.base_features:
            base_features = self.multi_model.base_features
        else:
            # Fallback to default base features
            base_features = [
                'common_projects', 'common_departments', 'common_subdepartments',
                'dept_wins', 'total_win_value', 'win_rate', 'price_range_overlap',
                'recent_activity_score', 'dept_engagement_score', 'subdept_specialization'
            ]
        
        # Initialize weight history
        for feature in base_features:
            self.weight_history[feature] = []
        
        for seed in range(num_seeds):
            print(f"\nRun {seed+1}/{num_seeds} with seed {seed}")
            
            # Create model with specific seed
            temp_model = MultiReferenceGradientModel(self.db, random_seed=seed)
            temp_model.reference_companies = self.multi_model.reference_companies
            
            # Train model with the same polynomial settings as the original model
            use_poly = self.multi_model.use_polynomial_features if hasattr(self.multi_model, 'use_polynomial_features') else False
            poly_degree = self.multi_model.poly_degree if hasattr(self.multi_model, 'poly_degree') else 2
            
            success = await temp_model.train_unified_model(use_polynomial_features=use_poly, degree=poly_degree)
            if not success:
                print(f"Failed to train model for seed {seed}")
                continue
            
            # Record weights
            weights = temp_model.model.get_feature_weights()
            
            # Focus on base features if we're using polynomial features
            if len(weights) > len(base_features):
                # Just look at the original features (not interactions)
                for feature in base_features:
                    if feature in weights:
                        if feature not in self.weight_history:
                            self.weight_history[feature] = []
                        self.weight_history[feature].append(weights[feature])
            else:
                # Record all weights
                for feature, weight in weights.items():
                    if feature not in self.weight_history:
                        self.weight_history[feature] = []
                    self.weight_history[feature].append(weight)
            
            # Record rankings for each reference company
            for ref in temp_model.reference_companies:
                company_name = ref['name']
                if company_name not in self.rank_stability:
                    self.rank_stability[company_name] = {}
                    
                # Apply model to this company
                target_data = await temp_model.apply_to_new_target(ref['target'], ref['name'])
                
                if target_data is None:
                    print(f"Failed to apply model to {company_name} with seed {seed}")
                    continue
                
                # Track known competitor ranks
                for tin in ref['competitors']:
                    comp_match = target_data[target_data['tin'] == tin]
                    if not comp_match.empty:
                        if tin not in self.rank_stability[company_name]:
                            self.rank_stability[company_name][tin] = []
                        
                        rank = comp_match['rank'].iloc[0]
                        self.rank_stability[company_name][tin].append(rank)
        
        # Analyze results
        self._analyze_weight_stability()
        self._analyze_rank_stability()
        
        # Visualize results
        self._visualize_stability_results()
    
    def _analyze_weight_stability(self):
        """Analyze the stability of feature weights across random seeds."""
        print("\n" + "="*80)
        print("FEATURE WEIGHT STABILITY ANALYSIS")
        print("="*80)
        
        # Calculate statistics for each feature
        weight_stats = {}
        for feature, weights in self.weight_history.items():
            if len(weights) <= 1:
                continue
                
            mean = np.mean(weights)
            std = np.std(weights)
            cv = (std / abs(mean)) * 100 if abs(mean) > 1e-10 else float('inf')
            
            weight_stats[feature] = {
                'mean': mean,
                'std': std,
                'cv': cv,
                'min': np.min(weights),
                'max': np.max(weights),
                'range': np.max(weights) - np.min(weights),
                'consistent_sign': (np.min(weights) > 0) or (np.max(weights) < 0)
            }
        
        # Sort by coefficient of variation (CV)
        sorted_stats = sorted(weight_stats.items(), key=lambda x: x[1]['cv'])
        
        # Print table
        print("\nFeature weight stability (sorted by coefficient of variation):")
        table_data = []
        for feature, stats in sorted_stats:
            table_data.append([
                feature,
                f"{stats['mean']:.4f}",
                f"{stats['std']:.4f}",
                f"{stats['cv']:.1f}%",
                f"{stats['min']:.4f} to {stats['max']:.4f}",
                "Yes" if stats['consistent_sign'] else "No"
            ])
        
        print(tabulate(
            table_data,
            headers=['Feature', 'Mean', 'Std Dev', 'CV', 'Range', 'Consistent Sign'],
            tablefmt='grid'
        ))
        
        # Summary metrics
        consistent_sign_pct = sum(1 for _, s in sorted_stats if s['consistent_sign']) / len(sorted_stats) * 100
        high_cv_pct = sum(1 for _, s in sorted_stats if s['cv'] > 50) / len(sorted_stats) * 100
        
        print(f"\nStability summary:")
        print(f"  Features with consistent sign: {consistent_sign_pct:.1f}%")
        print(f"  Features with high variation (CV > 50%): {high_cv_pct:.1f}%")
        
        if consistent_sign_pct >= 80:
            print("  OVERALL ASSESSMENT: Feature signs are STABLE")
        elif consistent_sign_pct >= 50:
            print("  OVERALL ASSESSMENT: Feature signs are MODERATELY STABLE")
        else:
            print("  OVERALL ASSESSMENT: Feature signs are UNSTABLE")
    
    def _analyze_rank_stability(self):
        """Analyze the stability of competitor ranks across random seeds."""
        print("\n" + "="*80)
        print("COMPETITOR RANK STABILITY ANALYSIS")
        print("="*80)
        
        all_rank_cvs = []
        
        for company, competitors in self.rank_stability.items():
            print(f"\n{company}:")
            
            # Calculate statistics for each competitor
            rank_stats = []
            for tin, ranks in competitors.items():
                if len(ranks) <= 1:
                    continue
                    
                mean = np.mean(ranks)
                std = np.std(ranks)
                cv = (std / mean) * 100 if mean > 0 else float('inf')
                
                rank_stats.append((tin, mean, std, cv, min(ranks), max(ranks)))
                all_rank_cvs.append(cv)
            
            # Sort by average rank
            rank_stats.sort(key=lambda x: x[1])
            
            # Print top 5 competitors
            table_data = []
            for tin, mean, std, cv, min_rank, max_rank in rank_stats[:5]:
                company_name = "Unknown"
                for ref in self.multi_model.reference_companies:
                    if ref['name'] == company:
                        is_known = "Yes" if tin in ref['competitors'] else "No"
                        break
                else:
                    is_known = "Unknown"
                
                table_data.append([
                    tin[:8] + "...",
                    f"{mean:.1f}",
                    f"{std:.1f}",
                    f"{cv:.1f}%",
                    f"{min_rank} to {max_rank}",
                    is_known
                ])
            
            print(f"Top 5 competitors by average rank:")
            print(tabulate(
                table_data,
                headers=['TIN', 'Avg Rank', 'Std Dev', 'CV', 'Rank Range', 'Known Competitor'],
                tablefmt='grid'
            ))
        
        # Overall rank stability
        avg_cv = np.mean(all_rank_cvs) if all_rank_cvs else float('inf')
        
        print(f"\nOverall rank stability:")
        print(f"  Average coefficient of variation: {avg_cv:.1f}%")
        
        if avg_cv < 20:
            print("  OVERALL ASSESSMENT: Competitor rankings are STABLE")
        elif avg_cv < 50:
            print("  OVERALL ASSESSMENT: Competitor rankings are MODERATELY STABLE")
        else:
            print("  OVERALL ASSESSMENT: Competitor rankings are UNSTABLE")
    
    def _visualize_stability_results(self):
        """Create visualizations of stability results."""
        # 1. Visualize weight stability for top features
        plt.figure(figsize=(12, 8))
        
        # Get features with sufficient data
        valid_features = {f: w for f, w in self.weight_history.items() if len(w) > 1}
        
        # Sort by variability (std/|mean|)
        feature_cv = {}
        for feature, weights in valid_features.items():
            mean = np.mean(weights)
            std = np.std(weights)
            cv = std / abs(mean) if abs(mean) > 1e-10 else float('inf')
            feature_cv[feature] = cv
        
        # Take top 10 most stable and top 10 most variable features
        sorted_features = sorted(feature_cv.items(), key=lambda x: x[1])
        stable_features = [f for f, _ in sorted_features[:min(10, len(sorted_features))]]
        unstable_features = [f for f, _ in sorted_features[-min(10, len(sorted_features)):]]
        
        # Prepare data for boxplot - focus on stable features
        boxplot_data = [self.weight_history[f] for f in stable_features]
        
        # Format feature names for display
        display_features = stable_features.copy()
        if any('_' in f for f in stable_features):
            # For polynomial features, make them more readable
            display_features = [f.replace(' ', '').replace('_', ' × ') for f in display_features]
            # Truncate if too long
            display_features = [f[:30] + '...' if len(f) > 30 else f for f in display_features]
        
        # Create boxplot
        plt.boxplot(
            boxplot_data,
            vert=False,
            labels=display_features,
            patch_artist=True,
            boxprops=dict(facecolor='lightgreen', alpha=0.8)
        )
        
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        plt.title('Most Stable Feature Weights Across Random Seeds', fontsize=14)
        plt.xlabel('Weight Value', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(self.output_dir, 'stable_weights.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Visualize rank stability for known competitors
        plt.figure(figsize=(12, 8))
        
        # Collect rank data for known competitors
        known_rank_data = {}
        known_names = {}
        
        for company, competitors in self.rank_stability.items():
            for ref in self.multi_model.reference_companies:
                if ref['name'] == company:
                    for tin in ref['competitors']:
                        if tin in competitors and len(competitors[tin]) > 1:
                            key = f"{company}: {tin[:6]}..."
                            known_rank_data[key] = competitors[tin]
                            known_names[key] = f"{company}"
        
        # Sort by median rank
        sorted_competitors = sorted(
            known_rank_data.items(),
            key=lambda x: np.median(x[1])
        )
        
        # Prepare data for boxplot
        boxplot_data = [ranks for _, ranks in sorted_competitors]
        labels = [name for name, _ in sorted_competitors]
        
        # Create boxplot
        plt.boxplot(
            boxplot_data,
            labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor='skyblue', alpha=0.8)
        )
        
        plt.title('Rank Stability for Known Competitors Across Random Seeds', fontsize=14)
        plt.ylabel('Rank', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(self.output_dir, 'rank_stability.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()


class BootstrapValidator:
    """Performs bootstrap analysis to assess feature importance stability."""
    
    def __init__(self, multi_model, db_conn):
        self.multi_model = multi_model
        self.db = db_conn
        self.bootstrap_weights = {}
        self.output_dir = 'results/bootstrap_output'
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def validate(self, n_iterations=100):
        """Perform bootstrap analysis."""
        print("\n" + "="*80)
        print("BOOTSTRAP FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Get combined dataset
        combined_data = await self.multi_model.build_combined_dataset()
        if combined_data is None:
            print("Failed to build dataset for bootstrapping")
            return
            
        # Get base features from the model if available
        if hasattr(self.multi_model, 'base_features') and self.multi_model.base_features:
            selected_features = self.multi_model.base_features
        else:
            # Fallback to default base features
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
        
        # Get polynomial settings
        use_poly = self.multi_model.use_polynomial_features if hasattr(self.multi_model, 'use_polynomial_features') else False
        poly_degree = self.multi_model.poly_degree if hasattr(self.multi_model, 'poly_degree') else 2
        
        # Initialize transformer if needed
        poly_transformer = None
        if use_poly:
            poly_transformer = PolynomialFeatures(degree=poly_degree, interaction_only=True, include_bias=False)
            # Fit on the entire dataset to get the same feature names
            X_all = combined_data[selected_features].values
            poly_transformer.fit(X_all)
            poly_feature_names = poly_transformer.get_feature_names_out(selected_features)
            
            # Initialize bootstrap weights for polynomial features
            for feature in poly_feature_names:
                self.bootstrap_weights[feature] = []
        else:
            # Initialize bootstrap weights for base features
            for feature in selected_features:
                self.bootstrap_weights[feature] = []
        
        n_samples = len(combined_data)
        
        # Run bootstrap iterations
        for i in range(n_iterations):
            if i % 10 == 0:
                print(f"Bootstrap iteration {i+1}/{n_iterations}")
                
            # Sample with replacement
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_sample = combined_data.iloc[bootstrap_indices].reset_index(drop=True)
            
            # Prepare features
            X = bootstrap_sample[selected_features].values
            y = bootstrap_sample['is_known_competitor'].values
            
            # Apply polynomial transformation if needed
            feature_names = selected_features
            if use_poly and poly_transformer is not None:
                X = poly_transformer.transform(X)
                feature_names = poly_transformer.get_feature_names_out(selected_features)
            
            # Initialize and train model
            temp_model = EnhancedGradientDescentClassifier(
                learning_rate=0.05, 
                iterations=2000,
                regularization=0.03,
                random_seed=np.random.randint(0, 10000)  # Random seed for each bootstrap
            )
            temp_model.fit(X, y, feature_names=feature_names)
            
            # Record weights
            weights = temp_model.get_feature_weights()
            for feature, weight in weights.items():
                if feature in self.bootstrap_weights:
                    self.bootstrap_weights[feature].append(weight)
        
        # Analyze bootstrap results
        self._analyze_bootstrap_results()
        
        # Visualize bootstrap results
        self._visualize_bootstrap_results()
    
    def _analyze_bootstrap_results(self):
        """Analyze bootstrap results and calculate confidence intervals."""
        print("\n" + "="*80)
        print("BOOTSTRAP ANALYSIS RESULTS")
        print("="*80)
        
        # Calculate confidence intervals and statistics
        bootstrap_stats = {}
        for feature, weights in self.bootstrap_weights.items():
            # Basic statistics
            mean = np.mean(weights)
            std = np.std(weights)
            
            # 95% confidence interval
            lower = np.percentile(weights, 2.5)
            upper = np.percentile(weights, 97.5)
            
            # Stability metrics
            stable_direction = (lower > 0 and upper > 0) or (lower < 0 and upper < 0)
            zero_crossings = sum(1 for i in range(len(weights)-1) if weights[i] * weights[i+1] < 0)
            
            bootstrap_stats[feature] = {
                'mean': mean,
                'std': std,
                'lower_ci': lower,
                'upper_ci': upper,
                'stable_direction': stable_direction,
                'zero_crossings': zero_crossings,
                'zero_cross_pct': zero_crossings / (len(weights)-1) * 100 if len(weights) > 1 else 0
            }
        
        # Sort by absolute mean value
        sorted_stats = sorted(bootstrap_stats.items(), key=lambda x: abs(x[1]['mean']), reverse=True)
        
        # Print table
        print("\nBootstrap analysis of feature importance (sorted by absolute mean):")
        table_data = []
        for feature, stats in sorted_stats:
            direction = "+" if stats['mean'] > 0 else "-"
            stability = "STABLE" if stats['stable_direction'] else "UNSTABLE"
            
            table_data.append([
                feature,
                f"{stats['mean']:.4f}",
                f"[{stats['lower_ci']:.4f}, {stats['upper_ci']:.4f}]",
                direction,
                stability,
                f"{stats['zero_cross_pct']:.1f}%"
            ])
        
        print(tabulate(
            table_data,
            headers=['Feature', 'Mean Weight', '95% CI', 'Direction', 'Stability', 'Zero Crossings'],
            tablefmt='grid'
        ))
        
        # Overall stability assessment
        stable_features = sum(1 for _, s in sorted_stats if s['stable_direction'])
        total_features = len(sorted_stats)
        stable_pct = (stable_features / total_features) * 100 if total_features > 0 else 0
        
        print(f"\nOverall stability assessment:")
        print(f"  Features with stable direction: {stable_features}/{total_features} ({stable_pct:.1f}%)")
        
        if stable_pct >= 80:
            print("  OVERALL ASSESSMENT: Feature importance is HIGHLY STABLE")
        elif stable_pct >= 60:
            print("  OVERALL ASSESSMENT: Feature importance is MODERATELY STABLE")
        else:
            print("  OVERALL ASSESSMENT: Feature importance is UNSTABLE")
    
    def _visualize_bootstrap_results(self):
        """Create visualizations of bootstrap results."""
        # 1. Visualize confidence intervals for top features
        plt.figure(figsize=(12, 8))
        
        # Sort features by absolute mean weight
        feature_means = {f: np.mean(w) for f, w in self.bootstrap_weights.items()}
        sorted_features = sorted(feature_means.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Take top 10 features
        top_features = [f for f, _ in sorted_features[:min(10, len(sorted_features))]]
        
        # Prepare data
        means = [np.mean(self.bootstrap_weights[f]) for f in top_features]
        lower_ci = [np.percentile(self.bootstrap_weights[f], 2.5) for f in top_features]
        upper_ci = [np.percentile(self.bootstrap_weights[f], 97.5) for f in top_features]
        
        # Format feature names for display
        display_features = top_features.copy()
        if any('_' in f for f in top_features):
            # For polynomial features, make them more readable
            display_features = [f.replace(' ', '').replace('_', ' × ') for f in display_features]
            # Truncate if too long
            display_features = [f[:30] + '...' if len(f) > 30 else f for f in display_features]
        
        # Calculate error bars
        yerr_lower = [m - l for m, l in zip(means, lower_ci)]
        yerr_upper = [u - m for m, u in zip(means, upper_ci)]
        
        # Plot
        y_pos = np.arange(len(display_features))
        bars = plt.barh(y_pos, means, xerr=[yerr_lower, yerr_upper], align='center', 
                alpha=0.7, capsize=5, color=['red' if m < 0 else 'blue' for m in means])
        
        plt.yticks(y_pos, display_features)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Feature Importance with 95% Confidence Intervals', fontsize=14)
        plt.xlabel('Weight Value', fontsize=12)
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(self.output_dir, 'bootstrap_confidence_intervals.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Visualize weight distributions for selected features
        plt.figure(figsize=(12, 10))
        
        # Choose features with most contrasting distributions
        # One highly positive, one highly negative, and one with high variance
        positive_feature = max(feature_means.items(), key=lambda x: x[1])[0]
        negative_feature = min(feature_means.items(), key=lambda x: x[1])[0]
        
        # Feature with highest variance around zero
        variance_around_zero = {
            f: np.var(w) / (abs(np.mean(w)) + 1e-10) 
            for f, w in self.bootstrap_weights.items()
        }
        variable_feature = max(variance_around_zero.items(), key=lambda x: x[1])[0]
        
        # Create a 3x1 subplot
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot histograms
        for i, (feature, title) in enumerate([
            (positive_feature, 'Strongest Positive Feature'),
            (negative_feature, 'Strongest Negative Feature'),
            (variable_feature, 'Most Variable Feature')
        ]):
            weights = self.bootstrap_weights[feature]
            axs[i].hist(weights, bins=30, alpha=0.7, color='skyblue')
            axs[i].axvline(x=0, color='red', linestyle='--')
            axs[i].axvline(x=np.mean(weights), color='black', linestyle='-', label='Mean')
            axs[i].axvline(x=np.percentile(weights, 2.5), color='green', linestyle=':', label='2.5%')
            axs[i].axvline(x=np.percentile(weights, 97.5), color='green', linestyle=':', label='97.5%')
            
            # Nice formatted feature name for title
            display_name = feature
            if '_' in feature:
                display_name = feature.replace('_', ' × ')
            
            axs[i].set_title(f"{title}: {display_name}", fontsize=12)
            axs[i].set_xlabel('Weight Value')
            axs[i].set_ylabel('Frequency')
            
            # Only add legend to first plot
            if i == 0:
                axs[i].legend()
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(self.output_dir, 'bootstrap_distributions.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()


async def main():
    """Main entry point for the script."""
    start_time = datetime.now()
    logger.info("Starting Enhanced Multi-Reference Gradient Model")
    
    # Initialize database connection
    db = DbConnection()
    await db.initialize()
    
    try:
        # Set master random seed
        np.random.seed(42)
        random.seed(42)
        
        # Create multi-reference model
        model = MultiReferenceGradientModel(db, random_seed=42)
        
        print("\n1. TRAINING UNIFIED BASE MODEL")
        # Train the base model first (no polynomial features for simplicity)
        await model.train_unified_model(use_polynomial_features=False)
        model.analyze_feature_weights()
        
        print("\n2. PERFORMING CROSS-COMPANY VALIDATION")
        # Perform cross-company validation
        validator = CrossCompanyValidator(model, db)
        await validator.validate()
        
        print("\n3. CHECKING STABILITY ACROSS RANDOM SEEDS")
        # Check stability across random seeds
        seed_validator = RandomSeedValidator(model, db)
        await seed_validator.validate(num_seeds=5)  # Reduced for performance
        
        print("\n4. RUNNING BOOTSTRAP FEATURE IMPORTANCE ANALYSIS")
        # Bootstrap feature importance analysis
        bootstrap_validator = BootstrapValidator(model, db)
        await bootstrap_validator.validate(n_iterations=50)  # Reduced for performance
        
        # For polynomial features, try a safer approach with proper error handling
        try:
            print("\n5. TRAINING FINAL ENHANCED MODEL WITH POLYNOMIAL FEATURES")
            # Train enhanced model with polynomial features
            enhanced_model = MultiReferenceGradientModel(db, random_seed=42)
            await enhanced_model.train_unified_model(use_polynomial_features=True, degree=2)
            enhanced_model.analyze_feature_weights()
            
            # Only run validation on polynomial model if it trained successfully
            print("\n6. VALIDATING POLYNOMIAL MODEL")
            poly_validator = CrossCompanyValidator(enhanced_model, db)
            await poly_validator.validate()
            
            # Apply the final model to a new target (optional)
            # new_target = "0107545000217"  # Example: PowerLine Engineering
            # await enhanced_model.apply_to_new_target(new_target, "PowerLine Engineering")
        except Exception as e:
            logger.error(f"Error in polynomial model phase: {str(e)}")
            print("\nSkipping polynomial features due to error. This is likely due to the feature name format issue.")
            print("You can modify the code to handle polynomial feature names differently or skip this step.")
        
        logger.info("Enhanced multi-reference gradient model analysis complete")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
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