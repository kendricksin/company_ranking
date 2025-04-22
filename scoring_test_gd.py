#!/usr/bin/env python3
"""
Enhanced Competitor Scoring Validator

This script evaluates and compares multiple scoring methods for identifying strong competitors
in Thai government project bidding. It implements:

1. The original/current scoring formula
2. The enhanced scoring formula from the previous test
3. A new balanced scoring approach with additional metrics
4. A gradient descent-based logistic regression score trained on known competitors

The script validates these scoring methods against known strong competitors
and provides detailed comparative analysis and visualizations.
"""

import os
import sys
import asyncio
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import asyncpg
from tabulate import tabulate
from typing import Dict, List, Any, Set, Optional, Tuple, Union
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler # Added StandardScaler
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scoring_validation.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class SimpleGradientDescentClassifier:
    """Simple Logistic Regression Classifier trained with Gradient Descent."""
    def __init__(self, learning_rate=0.01, iterations=1000, regularization=0.01):
        self.lr = learning_rate
        self.iter = iterations
        self.regularization = regularization # Added L2 regularization
        self.weights = None
        self.bias = None # Changed from 0 to None for clarity
        self.scaler = StandardScaler() # Use StandardScaler for feature scaling

    def sigmoid(self, z):
        # Clip input to avoid overflow in exp
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # --- Feature Scaling ---
        X_scaled = self.scaler.fit_transform(X)

        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.iter):
            linear_model = np.dot(X_scaled, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Compute gradients with L2 regularization
            dw = (1/n_samples) * (np.dot(X_scaled.T, (y_pred - y)) + self.regularization * self.weights) # Added regularization term
            db = (1/n_samples) * np.sum(y_pred - y)

            # Update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        """Predicts probabilities (sigmoid output)."""
        if self.weights is None or self.bias is None:
            raise Exception("Model has not been trained yet.")

        # --- Feature Scaling ---
        X_scaled = self.scaler.transform(X) # Use transform, not fit_transform

        linear_model = np.dot(X_scaled, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """Predicts class labels based on threshold."""
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= threshold, 1, 0)


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
    async def read_sql_async(self, query: str, params: List[Any] = None) -> pd.DataFrame:
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


class EnhancedScoringValidator:
    """Class to validate and compare multiple competitor scoring methodologies."""

    def __init__(self, db: DbConnection):
        self._db = db
        self.output_dir = "scoring_results"
        self.model = None # Attribute to hold the trained gradient descent model

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    async def analyze_competitors(
        self,
        target_company_tin: str,
        known_competitors: List[str],
        validation_set: Optional[List[str]] = None,
        lookback_months: int = 60
    ) -> Dict:
        """
        Comprehensive analysis of competitors using multiple scoring methods.
        Args:
            target_company_tin: TIN of the target company
            known_competitors: List of TINs for known strong competitors (training set)
            validation_set: Optional list of TINs for validation (test set)
            lookback_months: Number of months to look back for project data

        Returns: 
            Dictionary with analysis results
        """
        logger.info(f"Starting competitor analysis for target company {target_company_tin}")
        start_time = time.time()

        # Get target company information
        target_company = await self._get_company_info(target_company_tin)
        if not target_company:
            logger.error(f"Target company {target_company_tin} not found")
            return {"error": "Target company not found"}

        logger.info(f"Target company: {target_company['name']} ({target_company_tin})")

        # Get project data for the target company
        target_projects = await self._get_company_projects(
            target_company_tin, lookback_months=lookback_months
        )
        logger.info(f"Target company has {len(target_projects)} projects in the last {lookback_months} months")

        # Get all potential competitors
        all_competitors = await self._get_all_competitors(
            target_company_tin, lookback_months=lookback_months
        )
        logger.info(f"Found {len(all_competitors)} potential competitors")

        if all_competitors.empty:
            logger.error("No competitors found for analysis")
            return {"error": "No competitors found"}

        # Calculate additional metrics for all competitors
        all_competitors_enhanced = await self._calculate_all_competitor_metrics(
            target_company_tin, target_company, target_projects, all_competitors
        )

        # --- Train Gradient Descent Model --- 
        self.model = await self._train_gradient_model(all_competitors_enhanced, known_competitors, target_company)

        # --- Define and calculate different scoring methods --- 
        scoring_methods = {
            'current_score': all_competitors_enhanced['current_score'],
            'enhanced_score': all_competitors_enhanced['enhanced_score'],
            'balanced_score': self._calculate_balanced_score(all_competitors_enhanced),
            'weighted_score': self._calculate_weighted_score(all_competitors_enhanced),
            'logistic_score': self._calculate_logistic_score(all_competitors_enhanced),
            'gradient_score': self._calculate_gradient_score(all_competitors_enhanced) # Added Gradient Score 
        }

        # Add scores to the DataFrame
        for method, scores in scoring_methods.items():
             # Ensure the method exists (e.g., gradient_score might fail if training failed)
            if scores is not None and not scores.empty:
                all_competitors_enhanced[method] = scores
            else:
                logger.warning(f"Could not calculate scores for method: {method}. Skipping.")


        # Evaluate the scoring methods
        evaluation_results = self._evaluate_scoring_methods(
            all_competitors_enhanced, known_competitors, validation_set
        )

        # Perform detailed analysis of target competitors
        target_competitors_analysis = await self._analyze_target_competitors(
            target_tin=target_company_tin,
            target_company=target_company,
            target_tins=known_competitors,
            # all_competitors_enhanced=all_competitors_enhanced,
            competitors_df=all_competitors
        )

        # Create visualizations
        self._create_visualizations(
            all_competitors_enhanced, known_competitors, validation_set
        )

        # Prepare and save results
        results = {
            'target_company': target_company,
            'competitors_count': len(all_competitors_enhanced),
            'known_competitors_count': len(known_competitors),
            'method_evaluation': evaluation_results,
            'target_competitors': target_competitors_analysis,
            'execution_time': time.time() - start_time
        }

        # Save full competitor data to CSV
        all_competitors_enhanced.to_csv(f"{self.output_dir}/all_competitors_analysis.csv", index=False)

        logger.info(f"Competitor analysis completed in {time.time() - start_time:.2f} seconds")

        return results

    async def _get_company_info(self, company_tin: str) -> Dict:
        """Get detailed company information including project statistics."""
        logger.info(f"Fetching company info for TIN: {company_tin}")

        company_query = """
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

        company_df = await self._db.read_sql_async(company_query, [company_tin])

        if company_df.empty:
            logger.warning(f"No company information found for TIN: {company_tin}")
            return None

        company_data = company_df.iloc[0].to_dict()

        # Calculate additional derived metrics
        company_data['win_rate'] = (
        (company_data['total_wins'] / company_data['total_projects']) * 100) if company_data['total_projects'] > 0 else 0.0
        company_data['active_years'] = (datetime.now() - company_data['first_project_date'].to_pydatetime()).days / 365.25 if not pd.isnull(company_data['first_project_date']) else 0
        company_data['activity_recency'] = (datetime.now() - company_data['last_project_date'].to_pydatetime()).days / 30 if not pd.isnull(company_data['last_project_date']) else float('inf')

        # Ensure unique_departments exists and is > 0
        company_data['unique_departments'] = company_data.get('unique_departments', 0)
        if company_data['unique_departments'] is None or company_data['unique_departments'] == 0:
            logger.warning(f"Target company {company_tin} has 0 unique departments. Setting to 1 for calculations.")
            company_data['unique_departments'] = 1 # Avoid division by zero

        return company_data

    async def _get_company_projects(self, company_tin: str, lookback_months: int = 60) -> pd.DataFrame:
        """Get detailed project data for a company with optional time limit."""
        logger.info(f"Fetching projects for company TIN: {company_tin} (lookback: {lookback_months} months)")

        time_limit = ""
        params = [company_tin]

        if lookback_months > 0:
            cutoff_date = datetime.now() - relativedelta(months=lookback_months)
            time_limit = "AND p.contract_date >= $2"
            params = [company_tin, cutoff_date]

        projects_query = f"""
        SELECT
            p.project_id,
            p.project_name,
            p.dept_name,
            p.dept_sub_name,
            p.project_type_name,
            p.purchase_method_name,
            p.purchase_method_group_name,
            p.sum_price_agree,
            p.price_build,
            p.contract_date,
            p.announce_date,
            p.province,
            b.bid,
            CASE WHEN p.winner_tin = b.tin THEN TRUE ELSE FALSE END as won
        FROM public_data.thai_project_bid_info b
        JOIN public_data.thai_govt_project p ON b.project_id = p.project_id
        WHERE b.tin = $1 {time_limit}
        ORDER BY p.contract_date DESC
        """

        projects_df = await self._db.read_sql_async(projects_query, params)

        if projects_df.empty:
            logger.warning(f"No projects found for company TIN: {company_tin}")

        return projects_df

    async def _get_all_competitors(self, company_tin: str, lookback_months: int = 60) -> pd.DataFrame:
            """Get all competitors for a target company with current scoring. [Corrected Query]"""
            logger.info(f"Fetching all competitors for company TIN: {company_tin}")

            time_limit_condition = ""
            params = [company_tin]

            if lookback_months > 0:
                # Use INTERVAL for PostgreSQL date calculations
                time_limit_condition = f"AND p.contract_date >= NOW() - INTERVAL '{lookback_months} months'"
                # Note: We construct the interval string directly here.
                # asyncpg generally prefers passing parameters separately, but intervals
                # are often easier to construct directly in the query string unless dynamic.
                # If using parameters, it would look like:
                # time_limit_condition = "AND p.contract_date >= NOW() - INTERVAL $2"
                # params.append(f"{lookback_months} months")
                # Let's stick to direct string construction for interval as it's less prone to type issues here.

            # Construct the query using f-string (carefully, as params are separate)
            # We apply the time limit condition string directly into the query parts where needed.
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
                    ) as last_activity_date
                FROM direct_competitors dc
            )
            -- Apply final calculations and sorting
            SELECT
                tin,
                name,
                common_projects,
                common_departments,
                total_projects,
                total_wins,
                dept_wins,
                avg_bid_amount,
                last_activity_date,
                -- Derived fields
                CASE WHEN total_projects > 0 THEN ROUND((total_wins::numeric / total_projects) * 100, 2) ELSE 0 END as win_rate,
                -- Calculate months since last activity relative to NOW()
                EXTRACT(DAYS FROM (NOW() - COALESCE(last_activity_date, NOW() - INTERVAL '{lookback_months + 1} months'))) / 30.0 as months_since_last_activity, -- Use 30.0 for float division
                -- Current competitive strength score (0-100)
                GREATEST(0, LEAST(100,
                    (common_projects * 60.0 / GREATEST(1, total_projects)) +
                    -- Use GREATEST(1, dept_wins) in denominator if common_departments can be 0 but dept_wins > 0? Or GREATEST(1, common_departments) is safer.
                    (dept_wins * 20.0 / GREATEST(1, common_departments)) +
                    (total_wins * 20.0 / GREATEST(1, total_projects))
                )) as current_score
            FROM competitor_stats
            -- Ensure total_projects is positive to avoid division by zero in score calculation and relevance filter
            WHERE total_projects > 0
            ORDER BY
                common_projects DESC,
                current_score DESC
            """

            # Execute with only the company TIN as parameter, as the interval is now part of the query string
            competitors_df = await self._db.read_sql_async(competitors_query, params)

            if competitors_df.empty:
                logger.warning(f"No competitors found for company TIN: {company_tin} within the lookback period.")

            return competitors_df

    async def _calculate_all_competitor_metrics(
            self, target_tin: str, target_company: Dict, target_projects: pd.DataFrame, competitors_df: pd.DataFrame
        ) -> pd.DataFrame:
            """Calculate comprehensive metrics for all competitors. [Corrected Type Error]"""
            logger.info(f"Calculating comprehensive metrics for {len(competitors_df)} competitors")

            if competitors_df.empty:
                return pd.DataFrame()

            # Create copy to avoid modifying the original
            result_df = competitors_df.copy()

            # Extract target company metrics for comparison
            target_departments = self._get_department_distribution(target_projects)
            target_dept_hhi = self._calculate_concentration(target_departments)
            target_types = self._get_project_type_distribution(target_projects)
            target_price_range = self._get_price_range(target_projects)
            target_unique_departments = target_company.get('unique_departments', 1) # Default to 1 if not found
            # Ensure target win rate is float
            target_win_rate_float = float(target_company.get('win_rate', 0.0))


            # Prepare for batch processing
            competitor_tins = result_df['tin'].tolist()
            total_competitors = len(competitor_tins)
            batch_size = 10 # Process 10 competitors at a time to avoid memory issues

            # Create progress dictionary for each metric
            metrics_data = {
                'tin': [],
                'common_subdepartments': [],
                'common_subdepartments_pct': [],
                'common_types': [],
                'common_types_pct': [],
                'price_range_overlap': [],
                'dept_concentration_similarity': [],
                'bid_pattern_similarity': [],
                'win_rate_similarity': [],
                'head_to_head_strength': [],
                'recent_activity_score': [],
                'project_value_similarity': [],
                'enhanced_score': [],
                'dept_overlap_pct': [] # Added for gradient descent feature
            }

            # Process in batches
            for i in range(0, total_competitors, batch_size):
                batch_tins = competitor_tins[i:min(i + batch_size, total_competitors)]
                logger.info(f"Processing batch {i//batch_size + 1}/{(total_competitors + batch_size - 1)//batch_size} ({len(batch_tins)} competitors)")

                for competitor_tin in batch_tins:
                    # Get competitor projects
                    competitor_projects = await self._get_company_projects(competitor_tin)

                    # Skip if competitor has no projects (can happen if lookback periods differ slightly)
                    # Or if competitor_row lookup fails
                    try:
                        competitor_row = result_df[result_df['tin'] == competitor_tin].iloc[0]
                    except IndexError:
                        logger.warning(f"Competitor TIN {competitor_tin} not found in initial competitor list after batching. Skipping.")
                        continue

                    if competitor_projects.empty:
                        # Still calculate metrics that don't rely on competitor_projects if needed,
                        # but many will be zero or default. For simplicity, skip if projects are empty.
                        logger.warning(f"Competitor {competitor_tin} has no projects in lookback period. Skipping metric calculation.")
                        # Add default values or skip appending if necessary - let's skip for now
                        continue


                    # Calculate common project IDs
                    common_project_ids = set(target_projects['project_id']) & set(competitor_projects['project_id'])

                    # Calculate subdepartment overlap
                    common_subdepts, subdept_overlap_pct = self._calculate_subdepartment_overlap(
                        target_projects, competitor_projects
                    )

                    # Calculate project type overlap
                    common_types, type_overlap_pct = self._calculate_project_type_overlap(
                        target_projects, competitor_projects
                    )

                    # Calculate price range overlap
                    competitor_price_range = self._get_price_range(competitor_projects)
                    price_range_overlap = self._calculate_range_overlap(
                        target_price_range, competitor_price_range
                    )

                    # Calculate department concentration (HHI)
                    competitor_departments = self._get_department_distribution(competitor_projects)
                    competitor_dept_hhi = self._calculate_concentration(competitor_departments)
                    dept_concentration_similarity = 1 - abs(target_dept_hhi - competitor_dept_hhi)

                    # Calculate bid pattern similarity
                    bid_ratio_similarity = await self._calculate_bid_pattern_similarity(
                        target_tin, competitor_tin
                    )

                    # Calculate win rate similarity --- <<< FIX IS HERE >>> ---
                    competitor_win_rate = competitor_row.get('win_rate', 0.0) # Use .get for safety
                    competitor_win_rate_float = float(competitor_win_rate) # Convert competitor win rate to float
                    win_rate_similarity = 1 - abs(target_win_rate_float - competitor_win_rate_float) / 100
                    # --- <<< END OF FIX >>> ---


                    # Calculate head-to-head strength
                    head_to_head_strength = await self._calculate_head_to_head_strength(
                        target_tin, competitor_tin
                    )

                    # Calculate recent activity score
                    recent_activity_score = await self._calculate_recent_activity_score(
                        target_tin, competitor_tin
                    )

                    # Calculate project value similarity
                    # Ensure target_company avg_project_value is float
                    target_avg_proj_val_float = float(target_company.get('avg_project_value', 0.0))
                    # Ensure competitor avg_bid_amount is float
                    competitor_avg_bid_float = float(competitor_row.get('avg_bid_amount', 0.0))
                    project_value_similarity = self._calculate_project_value_similarity(
                        target_avg_proj_val_float,
                        competitor_avg_bid_float
                    )

                    # Calculate Department Overlap Percentage (for gradient model)
                    # Ensure common_departments is int/float
                    common_dept_float = float(competitor_row.get('common_departments', 0))
                    dept_overlap_pct = (common_dept_float / max(1, target_unique_departments)) * 100


                    # Calculate enhanced score
                    enhanced_score = self._calculate_enhanced_score(
                        common_projects=len(common_project_ids),
                        total_projects=int(competitor_row.get('total_projects', 0)), # Ensure int
                        dept_overlap=dept_overlap_pct, # Use calculated percentage
                        subdept_overlap=subdept_overlap_pct,
                        price_range_overlap=price_range_overlap,
                        bid_pattern_similarity=bid_ratio_similarity,
                        dept_concentration_similarity=dept_concentration_similarity,
                        win_rate_similarity=win_rate_similarity,
                        type_overlap=type_overlap_pct
                    )

                    # Store the results
                    metrics_data['tin'].append(competitor_tin)
                    metrics_data['common_subdepartments'].append(common_subdepts)
                    metrics_data['common_subdepartments_pct'].append(subdept_overlap_pct)
                    metrics_data['common_types'].append(common_types)
                    metrics_data['common_types_pct'].append(type_overlap_pct)
                    metrics_data['price_range_overlap'].append(price_range_overlap)
                    metrics_data['dept_concentration_similarity'].append(dept_concentration_similarity)
                    metrics_data['bid_pattern_similarity'].append(bid_ratio_similarity)
                    metrics_data['win_rate_similarity'].append(win_rate_similarity)
                    metrics_data['head_to_head_strength'].append(head_to_head_strength)
                    metrics_data['recent_activity_score'].append(recent_activity_score)
                    metrics_data['project_value_similarity'].append(project_value_similarity)
                    metrics_data['enhanced_score'].append(enhanced_score)
                    metrics_data['dept_overlap_pct'].append(dept_overlap_pct) # Store dept overlap pct

            # Convert to DataFrame
            metrics_df = pd.DataFrame(metrics_data)

            # Merge with original competitor data
            if not metrics_df.empty:
                # Make sure 'tin' columns are of the same type before merging
                result_df['tin'] = result_df['tin'].astype(str)
                metrics_df['tin'] = metrics_df['tin'].astype(str)
                result_df = pd.merge(result_df, metrics_df, on='tin', how='left')
            else:
                # If metrics_df is empty, add empty columns to result_df to avoid errors later
                missing_cols = [col for col in metrics_data.keys() if col != 'tin' and col not in result_df.columns]
                for col in missing_cols:
                    result_df[col] = pd.NA # Or appropriate default like 0 or 0.0


            # Fill NaNs that might arise from merging or calculations (important for modeling)
            numeric_cols = result_df.select_dtypes(include=np.number).columns.tolist()
            for col in numeric_cols:
                # Only fill NaNs for the newly calculated metrics (avoid overwriting existing scores yet)
                if col in metrics_data.keys() and col != 'tin':
                    if result_df[col].isnull().any():
                        logger.warning(f"Column '{col}' has NaNs after metric calculation/merge. Filling with 0.")
                        result_df[col] = result_df[col].fillna(0)


            return result_df

    def _get_department_distribution(self, projects_df: pd.DataFrame) -> Dict[str, int]:
        """Get distribution of projects by department."""
        if projects_df.empty:
            return {}

        return projects_df.groupby('dept_name').size().to_dict()

    def _get_project_type_distribution(self, projects_df: pd.DataFrame) -> Dict[str, int]:
        """Get distribution of projects by type."""
        if projects_df.empty:
            return {}

        return projects_df.groupby('project_type_name').size().to_dict()

    def _get_price_range(self, projects_df: pd.DataFrame) -> Tuple[float, float]:
        """Get the price range (min, max) from projects."""
        if projects_df.empty or projects_df['sum_price_agree'].isnull().all():
            return (0, 0)

        min_price = projects_df['sum_price_agree'].min()
        max_price = projects_df['sum_price_agree'].max()

        # Handle potential NaNs if not all are null
        min_price = 0 if pd.isnull(min_price) else min_price
        max_price = 0 if pd.isnull(max_price) else max_price

        return (min_price, max_price)


    def _calculate_subdepartment_overlap(
        self, target_projects: pd.DataFrame, competitor_projects: pd.DataFrame
    ) -> Tuple[int, float]:
        """Calculate overlap in subdepartments."""
        if target_projects.empty or competitor_projects.empty:
            return 0, 0.0

        # Combine dept and subdept
        target_subdepts = target_projects.groupby(['dept_name', 'dept_sub_name']).size().reset_index()
        competitor_subdepts = competitor_projects.groupby(['dept_name', 'dept_sub_name']).size().reset_index()

        if target_subdepts.empty or competitor_subdepts.empty:
            return 0, 0.0

        # Create full name for matching
        target_subdepts['dept_full'] = target_subdepts['dept_name'].astype(str) + ' - ' + target_subdepts['dept_sub_name'].astype(str)
        competitor_subdepts['dept_full'] = competitor_subdepts['dept_name'].astype(str) + ' - ' + competitor_subdepts['dept_sub_name'].astype(str)


        # Find common subdepartments
        common_subdepts = set(target_subdepts['dept_full']) & set(competitor_subdepts['dept_full'])

        # Calculate percentage
        overlap_pct = len(common_subdepts) / len(target_subdepts) * 100 if len(target_subdepts) > 0 else 0

        return len(common_subdepts), overlap_pct

    def _calculate_project_type_overlap(
        self, target_projects: pd.DataFrame, competitor_projects: pd.DataFrame
    ) -> Tuple[int, float]:
        """Calculate overlap in project types."""
        if target_projects.empty or competitor_projects.empty:
            return 0, 0.0

        target_types = set(target_projects['project_type_name'].dropna())
        competitor_types = set(competitor_projects['project_type_name'].dropna())

        common_types = target_types & competitor_types

        overlap_pct = len(common_types) / len(target_types) * 100 if len(target_types) > 0 else 0

        return len(common_types), overlap_pct

    def _calculate_range_overlap(self, range1: Tuple[float, float], range2: Tuple[float, float]) -> float:
        """Calculate the overlap between two ranges as a percentage of the first range."""
        min1, max1 = range1
        min2, max2 = range2

        # Handle edge cases
        if max1 <= min1 or max2 <= min2: # Check for valid ranges
             return 0.0

        # Ensure values are not NaN before comparison
        min1 = min1 if not pd.isnull(min1) else 0
        max1 = max1 if not pd.isnull(max1) else 0
        min2 = min2 if not pd.isnull(min2) else 0
        max2 = max2 if not pd.isnull(max2) else 0

        # Calculate overlap
        overlap_start = max(min1, min2)
        overlap_end = min(max1, max2)

        if overlap_start >= overlap_end:
            return 0.0

        overlap_length = overlap_end - overlap_start
        range1_length = max1 - min1

        return (overlap_length / range1_length) * 100 if range1_length > 0 else 0

    def _calculate_concentration(self, items_dict: Dict[str, int]) -> float:
        """Calculate Herfindahl-Hirschman Index (measure of concentration)."""
        if not items_dict:
            return 0

        total = sum(items_dict.values())
        if total == 0:
            return 0

        return sum((count / total) ** 2 for count in items_dict.values())

    async def _calculate_bid_pattern_similarity(self, target_tin: str, competitor_tin: str) -> float:
        """Calculate similarity in bidding patterns relative to reference price."""
        common_bids_query = """
        WITH target_bids AS (
            SELECT
                b.project_id,
                b.bid as target_bid
            FROM public_data.thai_project_bid_info b
            WHERE b.tin = $1
        ),
        competitor_bids AS (
            SELECT
                b.project_id,
                b.bid as competitor_bid
            FROM public_data.thai_project_bid_info b
            WHERE b.tin = $2
        )
        SELECT
            p.project_id,
            p.project_name,
            p.sum_price_agree,
            p.price_build,
            tb.target_bid,
            cb.competitor_bid,
            CASE WHEN p.winner_tin = $1 THEN TRUE ELSE FALSE END as target_won,
            CASE WHEN p.winner_tin = $2 THEN TRUE ELSE FALSE END as competitor_won
        FROM target_bids tb
        JOIN competitor_bids cb ON tb.project_id = cb.project_id
        JOIN public_data.thai_govt_project p ON tb.project_id = p.project_id
        WHERE p.price_build > 0 -- Only consider projects with valid reference price
        """

        common_bids_df = await self._db.read_sql_async(common_bids_query, [target_tin, competitor_tin])

        if common_bids_df.empty:
            return 0.0

        # Calculate bid pattern similarity
        bid_ratio_diffs = []
        for _, row in common_bids_df.iterrows():
            if row['price_build'] is not None and row['price_build'] > 0 and \
               row['target_bid'] is not None and row['competitor_bid'] is not None:
                target_ratio = row['target_bid'] / row['price_build']
                competitor_ratio = row['competitor_bid'] / row['price_build']
                bid_ratio_diffs.append(abs(target_ratio - competitor_ratio))


        if not bid_ratio_diffs:
            return 0.0

        avg_bid_ratio_diff = sum(bid_ratio_diffs) / len(bid_ratio_diffs)

        # Convert to similarity score (0-1)
        return max(0, 1 - min(avg_bid_ratio_diff, 1)) # Ensure similarity is between 0 and 1


    async def _calculate_head_to_head_strength(self, target_tin: str, competitor_tin: str) -> float:
        """Calculate how often the competitor wins when competing directly with the target."""
        query = """
        WITH common_projects AS (
            SELECT
                p.project_id,
                CASE
                    WHEN p.winner_tin = $1 THEN 1
                    WHEN p.winner_tin = $2 THEN -1
                    ELSE 0
                END AS outcome
            FROM public_data.thai_project_bid_info tb
            JOIN public_data.thai_project_bid_info cb ON tb.project_id = cb.project_id
            JOIN public_data.thai_govt_project p ON tb.project_id = p.project_id
            WHERE tb.tin = $1 AND cb.tin = $2
        )
        SELECT
            COUNT(*) AS total_common_projects,
            SUM(CASE WHEN outcome = -1 THEN 1 ELSE 0 END) AS competitor_wins,
            SUM(CASE WHEN outcome = 1 THEN 1 ELSE 0 END) AS target_wins,
            SUM(CASE WHEN outcome = 0 THEN 1 ELSE 0 END) AS no_winner
        FROM common_projects
        """

        result_df = await self._db.read_sql_async(query, [target_tin, competitor_tin])

        if result_df.empty or result_df.iloc[0]['total_common_projects'] == 0:
            return 0.0

        competitor_wins = result_df.iloc[0]['competitor_wins']
        total_common_projects = result_df.iloc[0]['total_common_projects']

        # Return the percentage of common projects where competitor wins
        return (competitor_wins / total_common_projects) * 100 if total_common_projects > 0 else 0

    async def _calculate_recent_activity_score(self, target_tin: str, competitor_tin: str, lookback_months: int = 24) -> float:
        """Calculate how recently the companies have competed against each other (scaled 0-1)."""
        cutoff_date = datetime.now() - relativedelta(months=lookback_months)
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
        params = [target_tin, competitor_tin, cutoff_date]
        result_df = await self._db.read_sql_async(query, params)

        if result_df.empty:
            return 0.0

        # Calculate recency score where more recent projects get higher weight
        recency_scores = []
        for _, row in result_df.iterrows():
            months_ago = row['months_ago']
            # Linear decay from 1.0 (now) to 0.0 (lookback_months ago)
            recency_score = max(0, 1 - (months_ago / lookback_months))
            recency_scores.append(recency_score)

        # Average of recency scores (0-1 scale)
        return sum(recency_scores) / len(recency_scores) if recency_scores else 0.0


    def _calculate_project_value_similarity(self, target_avg_value: float, competitor_avg_value: float) -> float:
        """Calculate similarity in project value ranges (0-1 scale)."""
        # Handle potential None values
        target_avg_value = target_avg_value if target_avg_value is not None else 0
        competitor_avg_value = competitor_avg_value if competitor_avg_value is not None else 0


        if target_avg_value <= 0 or competitor_avg_value <= 0:
            return 0.0

        # Ratio of smaller to larger value
        ratio = min(target_avg_value, competitor_avg_value) / max(target_avg_value, competitor_avg_value)

        # Convert to similarity score (0-1)
        return ratio

    def _calculate_enhanced_score(
        self,
        common_projects: int,
        total_projects: int,
        dept_overlap: float, # Expecting 0-100 scale
        subdept_overlap: float, # Expecting 0-100 scale
        price_range_overlap: float, # Expecting 0-100 scale
        bid_pattern_similarity: float, # Expecting 0-1 scale
        dept_concentration_similarity: float, # Expecting 0-1 scale
        win_rate_similarity: float, # Expecting 0-1 scale
        type_overlap: float # Expecting 0-100 scale
    ) -> float:
        """
        Calculate enhanced competitive strength score incorporating additional metrics.
        This formula gives weighted importance to:
        - Common projects (direct competition)
        - Department/subdepartment specialization overlap
        - Similar project value ranges
        - Similar bidding patterns
        - Similar project type focus
        """
        weights = {
            'common_projects': 30, # How often they directly compete (most important)
            'subdept_overlap': 20, # Specialization in same sub-departments
            'bid_pattern': 15,     # Similar bidding behavior
            'price_range': 10,     # Similar project price ranges
            'dept_overlap': 10,    # General department overlap
            'type_overlap': 10,    # Similar project types
            'concentration': 5     # Similar concentration/diversification strategy
        }

        # Calculate each component score (ensure all are 0-100)
        project_score = (common_projects / max(1, total_projects)) * 100
        subdept_overlap_score = subdept_overlap # Already 0-100
        bid_pattern_score = bid_pattern_similarity * 100 # Convert 0-1 to 0-100
        price_range_score = price_range_overlap # Already 0-100
        dept_overlap_score = dept_overlap # Already 0-100
        type_overlap_score = type_overlap # Already 0-100
        concentration_score = dept_concentration_similarity * 100 # Convert 0-1 to 0-100


        # Combine and normalize
        score = (
            (project_score * weights['common_projects']) +
            (subdept_overlap_score * weights['subdept_overlap']) +
            (bid_pattern_score * weights['bid_pattern']) +
            (price_range_score * weights['price_range']) +
            (dept_overlap_score * weights['dept_overlap']) +
            (type_overlap_score * weights['type_overlap']) +
            (concentration_score * weights['concentration'])
        ) / sum(weights.values()) # Normalize by total weight

        return min(100, max(0, score))

    def _calculate_balanced_score(self, competitors_df: pd.DataFrame) -> pd.Series:
        """
        Calculate balanced score with adjusted weights for better competitiveness assessment.
        This method: 
        1. Balances direct competition with domain similarity
        2. Emphasizes recent activity
        3. Gives more weight to project types and bidding patterns
        4. Properly handles cases with missing data
        """
        # Define weights for balanced score components
        weights = {
            'common_projects': 20,     # Direct competition indicator
            'total_projects': 5,       # Overall activity level
            'win_rate': 10,            # Success rate
            'subdept_overlap': 15,     # Specialization similarity
            'bid_pattern': 15, # Bidding behavior similarity
            'head_to_head': 10,        # Direct competition outcomes
            'recency': 10,             # Recent competition
            'dept_concentration': 10,  # Operational focus similarity
            'type_overlap': 5 # Project type similarity
        }

        # Ensure necessary columns exist and handle NaNs
        required_columns_map = {
            'common_projects': 'common_projects',
            'total_projects': 'total_projects',
            'win_rate': 'win_rate',
            'subdept_overlap': 'common_subdepartments_pct',
            'bid_pattern': 'bid_pattern_similarity',
            'head_to_head': 'head_to_head_strength',
            'recency': 'recent_activity_score',
            'dept_concentration': 'dept_concentration_similarity',
            'type_overlap': 'common_types_pct'
        }

        # Create a copy to avoid modifying original df
        df_copy = competitors_df.copy()

        # Convert numeric columns to float to avoid Decimal/float type conflicts
        numeric_cols = [
        'common_projects', 'total_projects', 'win_rate',
        'common_subdepartments_pct', 'bid_pattern_similarity',
        'head_to_head_strength', 'recent_activity_score',
        'dept_concentration_similarity', 'common_types_pct'
        ]
        df_copy[numeric_cols] = df_copy[numeric_cols].astype(float)


        component_scores = {}

        for key, col in required_columns_map.items():
            if col not in df_copy.columns:
                logger.warning(f"Column '{col}' for balanced score not found. Assigning score 0.")
                component_scores[key] = pd.Series(0, index=df_copy.index)
                df_copy[col] = 0 # Add column if missing
            else:
                 # Fill NaNs with 0 AFTER ensuring column exists
                 df_copy[col] = df_copy[col].fillna(0)

                 # Calculate component score based on metric type
                 if key == 'common_projects':
                         component_scores[key] = 100 * (df_copy[col].astype(float) / df_copy['total_projects'].astype(float).clip(lower=1))
                 elif key == 'total_projects':
                     # Activity factor (log scale, normalized 0-1)
                     max_log_projects = np.log1p(df_copy['total_projects'].max())
                     activity_factor = np.log1p(df_copy['total_projects']) / max_log_projects if max_log_projects > 0 else 0
                     component_scores[key] = 100 * activity_factor.clip(0, 1)
                 elif key == 'win_rate':
                      # Apply logistic transformation to emphasize middle range
                      component_scores[key] = 100 / (1 + np.exp(-0.1 * (df_copy[col].astype(float) - 50))) # Logistic curve centered at 50
                 elif key == 'subdept_overlap' or key == 'head_to_head' or key == 'type_overlap':
                      component_scores[key] = df_copy[col] # Already 0-100
                 elif key == 'bid_pattern' or key == 'recency' or key == 'dept_concentration':
                      component_scores[key] = 100 * df_copy[col] # Convert 0-1 to 0-100
                 else:
                     component_scores[key] = pd.Series(0, index=df_copy.index) # Default


        # Calculate final balanced score
        balanced_score = pd.Series(0.0, index=df_copy.index)
        total_weight = sum(weights.values())

        if total_weight > 0:
            for key, weight in weights.items():
                if key in component_scores:
                    balanced_score += component_scores[key] * weight

            balanced_score /= total_weight
        else:
             logger.warning("Total weight for balanced score is zero. Returning zeros.")


        # Clip values to 0-100 range
        return balanced_score.clip(0, 100)


    def _calculate_weighted_score(self, competitors_df: pd.DataFrame) -> pd.Series:
        """
        Calculate weighted score with dynamic weights based on data availability.
        This approach: 
        1. Automatically adjusts weights based on available metrics
        2. Gives higher weight to metrics with better data quality
        3. Focuses more on direct competition indicators
        """
        # Define base weights
        base_weights = {
            'common_projects': 30,
            'win_rate': 10,
            'subdept_overlap': 15,
            'bid_pattern': 15,
            'head_to_head': 10,
            'recency': 10,
            'type_overlap': 10
        }

        # Map base weight keys to actual column names
        column_map = {
            'common_projects': 'common_projects',
            'win_rate': 'win_rate',
            'subdept_overlap': 'common_subdepartments_pct',
            'bid_pattern': 'bid_pattern_similarity',
            'head_to_head': 'head_to_head_strength',
            'recency': 'recent_activity_score',
            'type_overlap': 'common_types_pct'
        }

        # Create a copy to work with
        df_copy = competitors_df.copy()

        numeric_cols = [
        'common_projects', 'total_projects', 'common_subdepartments_pct', 
        'win_rate', 'bid_pattern_similarity', 'recent_activity_score', 
        'head_to_head_strength'
        ]
        df_copy[numeric_cols] = df_copy[numeric_cols].astype(float)


        # Adjust weights based on available data and quality
        total_weight = 0
        adjusted_weights = {}
        data_quality = {}

        for key, base_weight in base_weights.items():
            col_name = column_map.get(key)
            quality = 0
            if col_name and col_name in df_copy.columns:
                # Ensure column is numeric before checking quality
                if pd.api.types.is_numeric_dtype(df_copy[col_name]):
                    # Calculate quality as fraction of non-null, non-zero values (for some metrics 0 is meaningful)
                    # For simplicity, just check non-null for now
                    quality = df_copy[col_name].notna().mean()
                    # Fill NaNs after quality check
                    df_copy[col_name] = df_copy[col_name].fillna(0)

                else:
                    logger.warning(f"Column '{col_name}' for weighted score is not numeric. Skipping quality check.")
                    df_copy[col_name] = 0 # Treat non-numeric as unavailable


            data_quality[col_name] = quality
            adjusted_weight = base_weight * quality # Weight adjusted by data quality
            adjusted_weights[col_name] = adjusted_weight
            total_weight += adjusted_weight


        # Normalize weights
        if total_weight > 0:
            for col_name in adjusted_weights:
                adjusted_weights[col_name] /= total_weight
        else:
            logger.warning("Total weight for weighted score is zero. Assigning equal weights if possible.")
            # Fallback: Assign equal weight if columns exist, otherwise 0
            num_available = sum(1 for col in column_map.values() if col in df_copy.columns)
            if num_available > 0:
                equal_weight = 1.0 / num_available
                for key, col_name in column_map.items():
                     adjusted_weights[col_name] = equal_weight if col_name in df_copy.columns else 0
            else:
                 # No columns available, all weights remain 0
                 pass



        # Calculate weighted score components
        weighted_score = pd.Series(0.0, index=df_copy.index)

        # Common Projects (requires total_projects)
        col_cp = column_map['common_projects']
        col_tp = 'total_projects' # Assumed available from earlier steps
        if col_cp in adjusted_weights and adjusted_weights[col_cp] > 0 and col_tp in df_copy:
             df_copy[col_tp] = df_copy[col_tp].fillna(0).clip(lower=1) # Handle NaN/zero for denominator
             cp_score = 100 * df_copy[col_cp] / df_copy[col_tp]
             weighted_score += cp_score.fillna(0) * adjusted_weights[col_cp]

        # Win Rate (0-100)
        col_wr = column_map['win_rate']
        if col_wr in adjusted_weights and adjusted_weights[col_wr] > 0:
             wr_score = df_copy[col_wr] # Already 0-100
             weighted_score += wr_score.fillna(0) * adjusted_weights[col_wr]


        # Subdepartment Overlap (0-100)
        col_subdept = column_map['subdept_overlap']
        if col_subdept in adjusted_weights and adjusted_weights[col_subdept] > 0:
            subdept_score = df_copy[col_subdept] # Already 0-100
            weighted_score += subdept_score.fillna(0) * adjusted_weights[col_subdept]


        # Bid Pattern Similarity (0-1 -> 0-100)
        col_bid = column_map['bid_pattern']
        if col_bid in adjusted_weights and adjusted_weights[col_bid] > 0:
            bid_score = 100 * df_copy[col_bid] # Convert 0-1 to 0-100
            weighted_score += bid_score.fillna(0) * adjusted_weights[col_bid]


        # Head-to-Head Strength (0-100)
        col_h2h = column_map['head_to_head']
        if col_h2h in adjusted_weights and adjusted_weights[col_h2h] > 0:
            h2h_score = df_copy[col_h2h] # Already 0-100
            weighted_score += h2h_score.fillna(0) * adjusted_weights[col_h2h]


        # Recent Activity Score (0-1 -> 0-100)
        col_rec = column_map['recency']
        if col_rec in adjusted_weights and adjusted_weights[col_rec] > 0:
             rec_score = 100 * df_copy[col_rec] # Convert 0-1 to 0-100
             weighted_score += rec_score.fillna(0) * adjusted_weights[col_rec]


        # Type Overlap (0-100)
        col_type = column_map['type_overlap']
        if col_type in adjusted_weights and adjusted_weights[col_type] > 0:
            type_score = df_copy[col_type] # Already 0-100
            weighted_score += type_score.fillna(0) * adjusted_weights[col_type]


        # Clip values to 0-100 range
        return weighted_score.clip(0, 100)


    def _calculate_logistic_score(self, competitors_df: pd.DataFrame) -> pd.Series:
        """
        Calculate logistic-transformed score to better differentiate similar competitors.

        This approach:
        1. Combines multiple factors with diminishing returns curve
        2. Adds non-linearity to create more differentiation
        3. Assigns competitive strength on a sigmoid curve
        """
        # Create copy
        df_copy = competitors_df.copy()

        numeric_cols = [
        'common_projects', 'total_projects', 'common_subdepartments_pct', 
        'win_rate', 'bid_pattern_similarity', 'recent_activity_score', 
        'head_to_head_strength'
        ]
        df_copy[numeric_cols] = df_copy[numeric_cols].astype(float)

        # Check for required columns and fill NaNs
        required_cols = {
            'common_projects': 'common_projects',
            'total_projects': 'total_projects',
            'subdept_factor': 'common_subdepartments_pct',
            'win_rate': 'win_rate',
            'bid_pattern': 'bid_pattern_similarity',
            'recency': 'recent_activity_score',
            'h2h': 'head_to_head_strength'
        }

        factors = {}
        for key, col in required_cols.items():
             if col not in df_copy.columns:
                 logger.warning(f"Column '{col}' for logistic score not found. Factor set to 0.")
                 df_copy[col] = 0 # Add if missing
                 factors[key] = pd.Series(0.0, index=df_copy.index)
             else:
                 # Fill NaNs before calculating factors
                 df_copy[col] = df_copy[col].fillna(0)

                 # Calculate factors based on key
                 if key == 'common_projects':
                      # Need total_projects for overlap ratio
                      df_copy['total_projects'] = df_copy['total_projects'].fillna(0).clip(lower=1)
                      factors['overlap_ratio'] = df_copy['common_projects'] / df_copy['total_projects']
                 elif key == 'subdept_factor':
                      factors[key] = df_copy[col].astype(float) / 100 # Convert 0-100 to 0-1
                 elif key == 'win_rate':
                      factors[key] = (df_copy[col].astype(float) / 100) # Convert 0-100 to 0-1
                 elif key == 'bid_pattern':
                      factors[key] = df_copy[col].astype(float) # Already 0-1
                 elif key == 'recency':
                      factors[key] = df_copy[col].astype(float) # Already 0-1
                 elif key == 'h2h':
                      factors[key] = df_copy[col].astype(float) / 100 # Convert 0-100 to 0-1


        # Ensure all expected factors exist, default to 0 if missing
        overlap_ratio = factors.get('overlap_ratio', pd.Series(0.0, index=df_copy.index))
        subdept_factor = factors.get('subdept_factor', pd.Series(0.0, index=df_copy.index))
        win_rate_factor = factors.get('win_rate', pd.Series(0.0, index=df_copy.index))
        bid_pattern_factor = factors.get('bid_pattern', pd.Series(0.0, index=df_copy.index))
        recency_factor = factors.get('recency', pd.Series(0.0, index=df_copy.index))
        h2h_factor = factors.get('h2h', pd.Series(0.0, index=df_copy.index))


        # Composite score calculation (weights from project.txt)
        composite_score = (
            overlap_ratio * 0.35 + # Direct competition
            subdept_factor * 0.2 + # Domain similarity
            win_rate_factor.astype(float) * 0.1 + # Success rate
            bid_pattern_factor * 0.15 + # Bidding behavior
            recency_factor * 0.1 + # Recent activity
            h2h_factor * 0.1 # Direct competition outcomes
        )

        # Apply sigmoid transformation
        k = 10 # Steepness
        x0 = 0.5 # Midpoint
        # Clip input to avoid overflow
        composite_clipped = np.clip(composite_score, -500, 500)
        logistic_score = 100 / (1 + np.exp(-k * (composite_clipped - x0)))

        return logistic_score.fillna(0) # Fill any potential NaNs resulting from calculation


    # --- New methods for Gradient Descent Score ---
    async def _train_gradient_model(self, competitors_df: pd.DataFrame, known_competitors: List[str], target_company: Dict) -> Optional[SimpleGradientDescentClassifier]:
        """Prepares data and trains the SimpleGradientDescentClassifier."""
        logger.info("Training Gradient Descent model...")

        # --- 1. Define Features --- 
        # Ensure dept_overlap_pct is calculated and available
        if 'dept_overlap_pct' not in competitors_df.columns:
             logger.warning("'dept_overlap_pct' not found. Calculating...")
             target_unique_departments = target_company.get('unique_departments', 1)
             if 'common_departments' in competitors_df.columns:
                 competitors_df['dept_overlap_pct'] = (competitors_df['common_departments'].fillna(0) / max(1, target_unique_departments)) * 100
             else:
                 logger.error("'common_departments' column missing. Cannot calculate 'dept_overlap_pct'. Training aborted.")
                 return None


        features = [
            'common_projects',
            'win_rate', # 0-100
            'bid_pattern_similarity', # 0-1
            'dept_overlap_pct', # Use the calculated percentage (0-100)
            'price_range_overlap', # 0-100
            'recent_activity_score' # 0-1
        ]

        # Check if all feature columns exist
        missing_features = [f for f in features if f not in competitors_df.columns]
        if missing_features:
            logger.error(f"Missing required features for training: {missing_features}. Training aborted.")
            return None

        # --- 2. Prepare Data ---
        df_train = competitors_df.copy()

        # Fill NaNs in feature columns (should have been done in _calculate_all...)
        for f in features:
             if df_train[f].isnull().any():
                 logger.warning(f"Feature '{f}' contains NaNs before training. Filling with 0.")
                 df_train[f] = df_train[f].fillna(0)


        X = df_train[features].values

        # --- 3. Create Labels ---
        df_train['label'] = 0
        df_train.loc[df_train['tin'].isin(known_competitors), 'label'] = 1
        y = df_train['label'].values

        if np.sum(y) == 0:
             logger.error("No known competitors found in the dataset (label=1). Cannot train model.")
             return None
        if np.sum(y) == len(y):
             logger.error("All competitors are marked as known competitors (label=1). Cannot train model.")
             return None

        # --- 4. Initialize and Train Model --- 
        try:
             model = SimpleGradientDescentClassifier(learning_rate=0.05, iterations=1500, regularization=0.05) # Adjusted params
             model.fit(X, y)
             logger.info("Gradient Descent model training complete.")

             # --- 5. Basic Validation --- 
             predictions = model.predict(X)
             accuracy = np.mean(predictions == y)
             # Accuracy on positive class (known competitors)
             known_mask = df_train['label'] == 1
             if np.sum(known_mask) > 0:
                 accuracy_known = np.mean(predictions[known_mask] == y[known_mask])
                 logger.info(f"Model Training Accuracy: {accuracy * 100:.2f}%, Accuracy on Known: {accuracy_known * 100:.2f}%")
             else:
                 logger.info(f"Model Training Accuracy: {accuracy * 100:.2f}% (No known competitors in training data for detailed check)")

             return model
        except Exception as e:
             logger.error(f"Error during model training: {str(e)}")
             logger.exception(e)
             return None


    # Sigmoid helper function 
    def sigmoid(self, z):
        # Clip input to avoid overflow in exp
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))


    def _calculate_gradient_score(self, competitors_df: pd.DataFrame) -> Optional[pd.Series]:
        """Calculates the competitor score using the trained gradient descent model."""
        if self.model is None:
            logger.error("Gradient descent model not trained. Cannot calculate gradient score.")
            return None # Return None if model isn't trained

        logger.info("Calculating Gradient Descent scores...")

        # --- Extract Features --- 
        features = [
            'common_projects',
            'win_rate',
            'bid_pattern_similarity',
            'dept_overlap_pct', # Ensure this matches training
            'price_range_overlap',
            'recent_activity_score'
        ]

        # Check if all feature columns exist
        missing_features = [f for f in features if f not in competitors_df.columns]
        if missing_features:
            logger.error(f"Missing required features for scoring: {missing_features}. Cannot calculate gradient score.")
            return None

        # Prepare data (handle NaNs)
        df_score = competitors_df.copy()
        for f in features:
             if df_score[f].isnull().any():
                 logger.warning(f"Feature '{f}' contains NaNs before scoring. Filling with 0.")
                 df_score[f] = df_score[f].fillna(0)

        X = df_score[features].values

        # --- Predict Probabilities --- 
        try:
             # Use predict_proba which directly gives the sigmoid output (0-1)
             scores = self.model.predict_proba(X)
             # Scale to 0-100
             gradient_scores = scores * 100
             return pd.Series(gradient_scores, index=competitors_df.index)
        except Exception as e:
            logger.error(f"Error during gradient score calculation: {str(e)}")
            logger.exception(e)
            return None

    # --- End of Gradient Descent Methods ---


    def _evaluate_scoring_methods(
        self, competitors_df: pd.DataFrame, known_competitors: List[str], validation_set: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate multiple scoring methods using various metrics.
        Args:
            competitors_df: DataFrame with competitor data and scores
            known_competitors: List of known strong competitor TINs (training set)
            validation_set: Optional list of validation TINs (test set)

        Returns:
            Dictionary with evaluation results
        """ 
        logger.info("Evaluating scoring methods")

        if competitors_df.empty:
            logger.warning("No competitor data available for evaluation")
            return {}

        # List of scoring methods to evaluate
        scoring_methods = [
            'current_score',
            'enhanced_score',
            'balanced_score',
            'weighted_score',
            'logistic_score',
            'gradient_score' # Added gradient score
        ]

        # Check which methods are available (have non-NaN scores)
        available_methods = []
        for method in scoring_methods:
             if method in competitors_df.columns and competitors_df[method].notna().any():
                 available_methods.append(method)
             else:
                  logger.warning(f"Scoring method '{method}' not available or all NaN. Excluding from evaluation.")


        if not available_methods:
            logger.error("No scoring methods found in the data for evaluation")
            return {}

        # Initialize results dictionary
        eval_results = {
            'total_competitors': len(competitors_df),
            'known_competitors': len(known_competitors),
            'metrics': {},
            'method_comparison': {}
        }

        # Calculate evaluation metrics for each method
        for method in available_methods:
            # Sort competitors by this method (handle NaNs by putting them last)
            sorted_df = competitors_df.sort_values(method, ascending=False, na_position='last').reset_index(drop=True)
            sorted_df[f'{method}_rank'] = sorted_df.index + 1

            # Get ranks of known competitors
            known_ranks = sorted_df[sorted_df['tin'].isin(known_competitors)][f'{method}_rank'].tolist()

            if not known_ranks:
                logger.warning(f"No known competitors found in the ranked data for method: {method}")
                # Assign default bad ranks if none found
                method_metrics = {
                    'avg_rank': float('inf'), 'median_rank': float('inf'),
                    'min_rank': float('inf'), 'max_rank': float('inf'),
                    'std_rank': 0, 'avg_percentile': 0,
                    'top_10': 0, 'top_20': 0, 'top_50': 0,
                    'ndcg_10': 0, 'ndcg_20': 0,
                    'precision_10': 0, 'precision_20': 0
                }

            else:
                 # Calculate percentiles
                 total_comps = len(sorted_df)
                 percentiles = [100 - ((rank -1) / total_comps * 100) for rank in known_ranks] # Rank starts at 1


                 # Calculate various metrics
                 method_metrics = {
                     'avg_rank': sum(known_ranks) / len(known_ranks),
                     'median_rank': np.median(known_ranks),
                     'min_rank': min(known_ranks),
                     'max_rank': max(known_ranks),
                     'std_rank': np.std(known_ranks),
                     'avg_percentile': sum(percentiles) / len(percentiles),
                     'top_10': sum(1 for r in known_ranks if r <= 10),
                     'top_20': sum(1 for r in known_ranks if r <= 20),
                      'top_50': sum(1 for r in known_ranks if r <= 50),
                     'ndcg_10': self._calculate_ndcg(sorted_df, known_competitors, k=10),
                     'ndcg_20': self._calculate_ndcg(sorted_df, known_competitors, k=20),
                     'precision_10': self._calculate_precision_at_k(sorted_df, known_competitors, k=10),
                     'precision_20': self._calculate_precision_at_k(sorted_df, known_competitors, k=20)
                 }

            # Store metrics
            eval_results['metrics'][method] = method_metrics

        # Compare methods
        if len(available_methods) > 1:
            # Calculate improvement over current score
            base_method = 'current_score' if 'current_score' in available_methods else available_methods[0]
            base_metrics = eval_results['metrics'].get(base_method)


            if base_metrics: # Ensure base metrics exist
                 for method in available_methods:
                     method_metrics = eval_results['metrics'].get(method)
                     if method != base_method and method_metrics: # Ensure method metrics exist
                         eval_results['method_comparison'][method] = {
                         'vs_method': base_method,
                             'avg_rank_improvement': base_metrics['avg_rank'] - method_metrics['avg_rank'],
                             'percentile_improvement': method_metrics['avg_percentile'] - base_metrics['avg_percentile'],
                             'ndcg_10_improvement': method_metrics['ndcg_10'] - base_metrics['ndcg_10'],
                             'precision_10_improvement': method_metrics['precision_10'] - base_metrics['precision_10']
                         }
            else:
                 logger.warning(f"Base method '{base_method}' metrics not found. Skipping method comparison.")


        # Calculate rank for each known competitor across methods
        competitor_analysis = []

        for tin in known_competitors:
            competitor_row = competitors_df[competitors_df['tin'] == tin]

            if competitor_row.empty:
                continue

            competitor_data = {
                'tin': tin,
                'name': competitor_row['name'].iloc[0] if 'name' in competitor_row else 'N/A', # Handle missing name
                'scores': {},
                'ranks': {}
            }

            for method in available_methods:
                 # Re-sort for each method to get rank accurately
                 sorted_df_eval = competitors_df.sort_values(method, ascending=False, na_position='last').reset_index(drop=True)
                 sorted_df_eval[f'{method}_rank'] = sorted_df_eval.index + 1

                 method_row = sorted_df_eval[sorted_df_eval['tin'] == tin]

                 if not method_row.empty:
                     score = method_row[method].iloc[0]
                     rank = method_row[f'{method}_rank'].iloc[0]
                     competitor_data['scores'][method] = float(score) if pd.notna(score) else None
                     competitor_data['ranks'][method] = int(rank) if pd.notna(rank) else None
                 else:
                      # Competitor might be missing if filtered out earlier
                      competitor_data['scores'][method] = None
                      competitor_data['ranks'][method] = None


            competitor_analysis.append(competitor_data)

        eval_results['competitor_analysis'] = competitor_analysis

        return eval_results

    def _calculate_ndcg(self, sorted_df: pd.DataFrame, relevant_tins: List[str], k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG) at rank k.
        This measures ranking quality when the order matters. 
        """
        # Get top k results
        top_k = sorted_df.head(k)

        # Calculate DCG
        dcg = 0
        for i, (_, row) in enumerate(top_k.iterrows(), 1):
            # Binary relevance: 1 if in relevant_tins, 0 otherwise
            rel = 1 if row['tin'] in relevant_tins else 0
            dcg += rel / np.log2(i + 1) # log2(1+1)=1, so first item has full weight

        # Calculate ideal DCG (IDCG)
        # In ideal case, all relevant items come first
        num_relevant_in_set = sum(1 for tin in relevant_tins if tin in sorted_df['tin'].values) # How many relevant items are actually in the dataset
        num_relevant_to_consider = min(num_relevant_in_set, k) # Consider at most k relevant items for ideal ranking

        idcg = sum(1 / np.log2(i + 1) for i in range(1, num_relevant_to_consider + 1))


        # Calculate NDCG
        return dcg / idcg if idcg > 0 else 0.0 # Handle case where IDCG is 0


    def _calculate_precision_at_k(self, sorted_df: pd.DataFrame, relevant_tins: List[str], k: int = 10) -> float:
        """
        Calculate precision at rank k.
        This measures what fraction of top k results are relevant.
        """
        # Get top k results
        top_k = sorted_df.head(k)

        # Count relevant items in top k
        relevant_in_top_k = sum(1 for _, row in top_k.iterrows() if row['tin'] in relevant_tins)

        # Calculate precision@k
        return relevant_in_top_k / k if k > 0 else 0.0


    async def _analyze_target_competitors(
        self, target_tin: str, target_company: Dict, target_tins: List[str], competitors_df: pd.DataFrame
    ) -> Dict:
        """Perform detailed analysis of target competitors."""
        logger.info(f"Analyzing {len(target_tins)} target competitors")

        # Fetch target projects once here:
        target_projects = await self._get_company_projects(target_tin)

        # Initialize results
        results = {'competitor_details': [], 'common_characteristics': {}, 'recommendations': {}}

        # Get subset of data for target competitors
        target_competitors_df = competitors_df[competitors_df['tin'].isin(target_tins)].copy()


        if target_competitors_df.empty:
            logger.warning("No target competitors found in the data")
            return results

        # Define available scoring methods dynamically
        available_methods = [
             m for m in ['current_score', 'enhanced_score', 'balanced_score', 'weighted_score', 'logistic_score', 'gradient_score']
             if m in target_competitors_df.columns
         ]


        # Analyze each target competitor
        target_projects = await self._get_company_projects(target_tin) # Fetch target projects once

        for _, competitor in target_competitors_df.iterrows():
            # Get detailed projects data
            competitor_projects = await self._get_company_projects(competitor['tin'])

            # Find common projects with target
            common_project_ids = set(target_projects['project_id']) & set(competitor_projects['project_id'])
            common_projects_df = target_projects[target_projects['project_id'].isin(common_project_ids)]

            # Extract key metrics
            competitor_detail = {
                'tin': competitor['tin'],
                'name': competitor.get('name', 'N/A'), # Use .get for safety
                'total_projects': int(competitor.get('total_projects', 0)),
                'total_wins': int(competitor.get('total_wins', 0)),
                'win_rate': float(competitor.get('win_rate', 0.0)),
                'common_projects': len(common_project_ids),
                'common_departments': int(competitor.get('common_departments', 0)),
                'scores': {
                     method: float(competitor[method]) if pd.notna(competitor.get(method)) else None
                     for method in available_methods # Use dynamic list
                 }, 
                'most_common_departments': self._get_top_items(competitor_projects, 'dept_name', 3),
                'most_common_project_types': self._get_top_items(competitor_projects, 'project_type_name', 3),
                'recent_common_projects': self._get_recent_common_projects(common_projects_df, 5)
            }

            results['competitor_details'].append(competitor_detail)

        # Analyze competitive strength indicators
        common_features = self._find_common_features(results['competitor_details'])

        results['common_characteristics'] = common_features

        # Develop recommendations based on analysis
        results['recommendations'] = self._generate_recommendations(
            target_company, results['competitor_details'], common_features, target_projects
        )

        return results

    def _get_top_items(self, df: pd.DataFrame, column: str, limit: int = 3) -> List[Dict]:
        """Get top items by frequency from a DataFrame column."""
        if df.empty or column not in df.columns:
            return []

        top_items = df[column].value_counts().head(limit)

        return [
            {'name': item, 'count': count}
            for item, count in top_items.items()
        ]

    def _get_recent_common_projects(self, common_projects_df: pd.DataFrame, limit: int = 5) -> List[Dict]:
        """Get most recent common projects."""
        if common_projects_df.empty:
            return []

        # Sort by contract date (most recent first)
        sorted_df = common_projects_df.sort_values('contract_date', ascending=False).head(limit)

        return [
            {
                'project_id': int(row['project_id']),
                'project_name': row['project_name'],
                'department': row['dept_name'],
                'contract_date': row['contract_date'].strftime('%Y-%m-%d') if pd.notna(row['contract_date']) else None,
                'value': float(row['sum_price_agree']) if pd.notna(row['sum_price_agree']) else None,
                'won': bool(row['won']) if pd.notna(row['won']) else None # Handle potential NaN in boolean column
            }
            for _, row in sorted_df.iterrows()
        ]


    def _find_common_features(self, competitor_details: List[Dict]) -> Dict:
        """Identify common features among strong competitors."""
        if not competitor_details:
            return {}

        # Extract features for analysis
        win_rates = [comp.get('win_rate', 0) for comp in competitor_details]
        project_counts = [comp.get('total_projects', 0) for comp in competitor_details]


        # Gather all departments and project types
        all_departments = []
        all_project_types = []

        for comp in competitor_details:
            if 'most_common_departments' in comp and comp['most_common_departments']:
                 all_departments.extend([dept.get('name', 'N/A') for dept in comp['most_common_departments']])

            if 'most_common_project_types' in comp and comp['most_common_project_types']:
                 all_project_types.extend([type_.get('name', 'N/A') for type_ in comp['most_common_project_types']])


        # Count frequencies
        dept_counts = {}
        for dept in all_departments:
            dept_counts[dept] = dept_counts.get(dept, 0) + 1

        type_counts = {}
        for type_ in all_project_types:
            type_counts[type_] = type_counts.get(type_, 0) + 1


        # Identify common features
        num_competitors = len(competitor_details)
        return {
            'avg_win_rate': np.mean(win_rates) if win_rates else 0,
            'min_win_rate': np.min(win_rates) if win_rates else 0,
            'max_win_rate': np.max(win_rates) if win_rates else 0,
            'avg_project_count': np.mean(project_counts) if project_counts else 0,
            'common_departments': [ 
                 dept for dept, count in dept_counts.items()
                 if count >= max(1, num_competitors / 2) # Appears in at least half, minimum 1
             ],
            'common_project_types': [
                type_ for type_, count in type_counts.items()
                if count >= max(1, num_competitors / 2) # Appears in at least half, minimum 1
             ]
        }


    def _generate_recommendations(
        self, target_company: Dict, competitor_details: List[Dict], common_features: Dict, target_projects: pd.DataFrame
    ) -> Dict:
        """Generate recommendations based on competitor analysis."""
        if not competitor_details or not common_features:
            return {}

        # Compare target with competitors
        target_win_rate = target_company.get('win_rate', 0)
        avg_competitor_win_rate = common_features.get('avg_win_rate', 0)

        win_rate_difference = target_win_rate - avg_competitor_win_rate

        # Generate insights
        insights = []

        if win_rate_difference < -10:
            insights.append(f"The target company's win rate ({target_win_rate:.1f}%) is significantly lower than the average ({avg_competitor_win_rate:.1f}%) of these strong competitors. This suggests a potential competitive disadvantage in bidding strategy or pricing.")
        elif win_rate_difference > 10:
            insights.append(f"The target company's win rate ({target_win_rate:.1f}%) is significantly higher than the average ({avg_competitor_win_rate:.1f}%) of these strong competitors. This indicates a competitive advantage that should be maintained.")


        # Check departmental focus
        target_departments_set = set(d['name'] for d in self._get_top_items(target_projects, 'dept_name', 100))        
        common_competitor_departments_set = set(common_features.get('common_departments', []))

        overlap_count = len(target_departments_set.intersection(common_competitor_departments_set))
        if len(target_departments_set) > 0 and overlap_count / len(target_departments_set) < 0.5:
             insights.append("The target company operates significantly in departments where these strong competitors are less common. Consider evaluating performance in departments where competitors are most active.")



        # Generate detailed recommendations
        recommendations = []

        # Win rate recommendations
        if target_win_rate < avg_competitor_win_rate:
            recommendations.append("Analyze bidding strategies of top competitors, particularly their bid-to-reference-price ratios in common projects.")
            recommendations.append("Investigate project types and departments where competitors consistently achieve high win rates.")


        # Department focus recommendations
        common_depts_list = common_features.get('common_departments')
        if common_depts_list:
             recommendations.append(f"Prioritize or strengthen presence in departments where strong competitors are most active: {', '.join(common_depts_list[:3])}.")


        # Project type recommendations
        common_types_list = common_features.get('common_project_types')
        if common_types_list:
             recommendations.append(f"Evaluate competitiveness in project types common among strong competitors: {', '.join(common_types_list[:3])}.")


        # Add a general recommendation
        recommendations.append("Continuously monitor the activities and performance of these key competitors using the enhanced scoring models.")


        return {
            'insights': insights,
            'recommendations': recommendations
        }

    def _create_visualizations(
        self, competitors_df: pd.DataFrame, known_competitors: List[str], validation_set: Optional[List[str]] = None
    ) -> None:
        """Create visualizations comparing scoring methods."""
        logger.info("Creating visualizations")

        if competitors_df.empty:
            logger.warning("No competitor data available for visualizations")
            return

        # Determine which scoring methods to visualize
        scoring_methods = [
            'current_score',
            'enhanced_score',
            'balanced_score',
            'weighted_score',
            'logistic_score',
            'gradient_score' # Added gradient score
        ]

        # Filter to methods that actually exist in the dataframe and have data
        available_methods = [
            method for method in scoring_methods
            if method in competitors_df.columns and competitors_df[method].notna().any()
        ]


        if len(available_methods) < 1:
             logger.warning("No scoring methods available for visualization.")
             return
        elif len(available_methods) < 2:
            logger.warning("Only one scoring method available. Skipping comparison visualizations.")
            # Proceed with single-method visualizations if applicable


        # Create output directory for visualizations
        viz_dir = f"{self.output_dir}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)

        # Create comparison scatter plots only if more than one method is available
        if len(available_methods) >= 2:
             base_method = 'current_score' if 'current_score' in available_methods else available_methods[0]
             for method in available_methods:
                 if method != base_method:
                    self._create_comparison_scatter(
                         competitors_df, base_method, method, known_competitors, validation_set, viz_dir
                     )


        # Create score distribution visualization (works with >= 1 method)
        self._create_score_distribution(
            competitors_df, available_methods, known_competitors, validation_set, viz_dir
        )

        # Create feature importance visualization (specific to gradient model or correlation)
        self._create_feature_importance(competitors_df, known_competitors, available_methods, viz_dir)


        # Create rank comparison visualization (works with >= 1 method)
        self._create_rank_comparison(competitors_df, known_competitors, available_methods, viz_dir)

    def _create_comparison_scatter(
        self,
        competitors_df: pd.DataFrame,
        base_method: str,
        compare_method: str,
        known_competitors: List[str],
        validation_set: Optional[List[str]],
        output_dir: str
    ) -> None:
        """Create scatter plot comparing two scoring methods."""
        try:
            # Create a copy of the data for visualization
            plot_data = competitors_df.copy()

            # Add a column to indicate known and validation competitors
            plot_data['is_known'] = plot_data['tin'].isin(known_competitors)
            plot_data['is_validation'] = plot_data['tin'].isin(validation_set if validation_set else []) # Handle None


            plt.figure(figsize=(10, 8))

            # Plot regular competitors
            regular_mask = ~plot_data['is_known'] & ~plot_data['is_validation'] 
            plt.scatter(
                plot_data.loc[regular_mask, base_method],
                plot_data.loc[regular_mask, compare_method],
                alpha=0.5, # Slightly adjusted alpha
                label='Other Competitors',
                color='grey', # Changed color
                s=30 # Reduced size
            )

            # Plot validation competitors if available
            if validation_set and plot_data['is_validation'].any():
                 plt.scatter(
                     plot_data.loc[plot_data['is_validation'], base_method],
                     plot_data.loc[plot_data['is_validation'], compare_method],
                     color='blue',
                     s=100, # Increased size
                     label='Validation Competitors',
                     marker='^', # Changed marker
                     edgecolors='black' # Added edge
                 )

            # Plot known strong competitors
            if plot_data['is_known'].any():
                 plt.scatter(
                     plot_data.loc[plot_data['is_known'], base_method],
                     plot_data.loc[plot_data['is_known'], compare_method],
                     color='red',
                     s=120, # Increased size
                     label='Known Strong Competitors',
                     marker='o',
                     edgecolors='black' # Added edge
                 )


            # Add labels for known competitors
            if 'name' in plot_data.columns: # Check if name column exists
                  for _, row in plot_data[plot_data['is_known']].iterrows():
                      plt.annotate(
                          row['name'],
                          (row[base_method], row[compare_method]),
                          xytext=(5, 5),
                          textcoords='offset points',
                          fontsize=8,
                          alpha=0.8
                      )


            # Add equality line
            min_val = min(plot_data[base_method].min(), plot_data[compare_method].min())
            max_val = max(plot_data[base_method].max(), plot_data[compare_method].max())
            if pd.notna(min_val) and pd.notna(max_val): # Check for NaN before plotting line
                 plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.4, label='y=x')


            plt.xlabel(f'{base_method.replace("_", " ").title()} Score') # Added "Score"
            plt.ylabel(f'{compare_method.replace("_", " ").title()} Score') # Added "Score"
            plt.title(f'Comparison: {base_method.replace("_", " ").title()} vs {compare_method.replace("_", " ").title()} Score') 
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save the plot
            plt.savefig(f'{output_dir}/{base_method}_vs_{compare_method}_scatter.png', dpi=300, bbox_inches='tight') 
            plt.close()

            logger.info(f"Created comparison scatter plot: {base_method} vs {compare_method}")

        except Exception as e:
            logger.error(f"Error creating comparison scatter plot ({base_method} vs {compare_method}): {str(e)}")
            logger.exception(e) # Log stack trace
            plt.close() # Ensure plot is closed on error


    def _create_score_distribution(
        self,
        competitors_df: pd.DataFrame,
        methods: List[str],
        known_competitors: List[str],
        validation_set: Optional[List[str]],
        output_dir: str
    ) -> None:
        """Create visualization of score distributions for different methods."""
        try:
            num_methods = len(methods)
            if num_methods == 0: return # Skip if no methods

            # Determine grid size
            ncols = min(num_methods, 2) # Max 2 columns
            nrows = (num_methods + ncols - 1) // ncols


            plt.figure(figsize=(6 * ncols, 5 * nrows))

            # For each scoring method, create a separate subplot
            for i, method in enumerate(methods):
                ax = plt.subplot(nrows, ncols, i + 1)

                # Create histogram/KDE of all scores (handle potential NaNs)
                scores = competitors_df[method].dropna()
                if scores.empty:
                     logger.warning(f"No valid scores for method '{method}' distribution plot.")
                     ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                     ax.set_title(f'{method.replace("_", " ").title()} Distribution')
                     continue # Skip to next method if no scores


                sns.histplot(scores, kde=True, color='skyblue', alpha=0.6, ax=ax, bins=15) # Adjusted color/bins

                # Mark known competitors
                known_scores = competitors_df[competitors_df['tin'].isin(known_competitors)][method].dropna()

                ymin, ymax = ax.get_ylim()
                if not known_scores.empty:
                     ax.vlines(known_scores, ymin=ymin, ymax=ymax * 0.9, color='red', linestyle='--', alpha=0.8, label='Known Competitors')



                # Mark validation competitors if available
                if validation_set:
                    validation_scores = competitors_df[competitors_df['tin'].isin(validation_set)][method].dropna()
                    if not validation_scores.empty:
                          ax.vlines(validation_scores, ymin=ymin, ymax=ymax * 0.9, color='blue', linestyle=':', alpha=0.8, label='Validation Competitors')


                ax.set_title(f'{method.replace("_", " ").title()} Distribution')
                ax.set_xlabel('Score')
                ax.set_ylabel('Frequency') # Changed label
                ax.set_ylim(ymin, ymax) # Reset ylim after vlines

                # Add legend only once if labels were added
                handles, labels = ax.get_legend_handles_labels()
                if handles: # Check if there are any labels to show
                    # Create a unique legend for the subplot
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')



            plt.tight_layout()
            plt.savefig(f'{output_dir}/score_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("Created score distribution visualization")

        except Exception as e:
            logger.error(f"Error creating score distribution visualization: {str(e)}")
            logger.exception(e)
            plt.close() # Ensure plot is closed on error


    def _create_feature_importance(
        self,
        competitors_df: pd.DataFrame,
        known_competitors: List[str], # known_competitors not used here, but kept for signature consistency
        methods: List[str], # Added methods list
        output_dir: str
    ) -> None:
        """Create feature importance visualization (correlation-based and model weights)."""
        try:
            # --- 1. Correlation-Based Importance ---
            feature_columns = [
                'common_projects', 'common_departments', 'dept_overlap_pct', # Added dept_overlap_pct
                'common_subdepartments_pct', 'common_types_pct',
                'price_range_overlap', 'bid_pattern_similarity',
                'dept_concentration_similarity', 'head_to_head_strength',
                'recent_activity_score', 'project_value_similarity',
                'win_rate', 'total_projects' # Added win_rate, total_projects
            ]

            # Filter to columns that exist and are numeric
            available_features = [
                 col for col in feature_columns
                 if col in competitors_df.columns and pd.api.types.is_numeric_dtype(competitors_df[col])
             ]


            if not available_features:
                logger.warning("No numeric feature columns available for correlation importance visualization")
                # Still try to plot model weights if possible
            else:
                 # Use available scoring methods from the list passed
                 available_scores = [m for m in methods if m in competitors_df.columns and pd.api.types.is_numeric_dtype(competitors_df[m])]

                 if not available_scores:
                     logger.warning("No numeric score columns available for correlation feature importance visualization")
                 else:
                     # Calculate correlation between features and scores
                     correlation_data = []
                     for score in available_scores:
                         for feature in available_features:
                             # Calculate correlation, handle potential errors (e.g., constant columns)
                             try:
                                  corr = competitors_df[feature].corr(competitors_df[score])
                                  correlation = abs(corr) if pd.notna(corr) else 0 # Use absolute correlation, handle NaN
                             except Exception:
                                  correlation = 0 # Default to 0 on error

                         correlation_data.append({
                                 'Feature': feature.replace('_', ' ').title(),
                                 'Score Method': score.replace('_', ' ').title(),
                                 'Correlation': correlation
                         })

                     # Convert to DataFrame for visualization
                     corr_df = pd.DataFrame(correlation_data)

                     if not corr_df.empty:
                          # Create heatmap
                          try:
                              plt.figure(figsize=(max(10, len(available_features) * 0.8), max(8, len(available_scores) * 0.6))) 
                              pivot_table = corr_df.pivot(index='Feature', columns='Score Method', values='Correlation')
                              sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.2f', linewidths=.5) 
                              plt.title('Absolute Correlation between Features and Scoring Methods')
                              plt.xticks(rotation=45, ha='right')
                              plt.tight_layout()
                              plt.savefig(f'{output_dir}/feature_correlation_heatmap.png', dpi=300, bbox_inches='tight') 
                              plt.close()
                              logger.info("Created feature correlation heatmap visualization")
                          except Exception as e_heatmap:
                              logger.error(f"Error creating correlation heatmap: {e_heatmap}")
                              plt.close()

                          # Create bar chart for top features per method (if multiple scores)
                          if len(available_scores) > 1:
                              try:
                                   ncols = min(len(available_scores), 2)
                                   nrows = (len(available_scores) + ncols - 1) // ncols
                                   plt.figure(figsize=(7 * ncols, 5 * nrows))

                                   for i, score in enumerate(available_scores):
                                       ax = plt.subplot(nrows, ncols, i + 1)
                                       score_title = score.replace('_', ' ').title()
                                       score_data = corr_df[corr_df['Score Method'] == score_title]
                                       score_data = score_data.sort_values('Correlation', ascending=False).head(10) # Top 10

                                       if not score_data.empty:
                                           sns.barplot(x='Correlation', y='Feature', data=score_data, palette='viridis', ax=ax) # Changed palette
                                           ax.set_title(f'Top Features Correlated with {score_title}')
                                           ax.set_xlabel('Absolute Correlation') # Clearer label
                                       else:
                                            ax.text(0.5, 0.5, 'No Correlation Data', ha='center', va='center', transform=ax.transAxes)
                                            ax.set_title(f'Top Features Correlated with {score_title}')

                                   plt.tight_layout() 
                                   plt.savefig(f'{output_dir}/feature_correlation_bars.png', dpi=300, bbox_inches='tight') # Changed filename
                                   plt.close()
                                   logger.info("Created feature correlation bar chart visualization")
                              except Exception as e_bars:
                                   logger.error(f"Error creating correlation bar charts: {e_bars}")
                                   plt.close()


            # --- 2. Gradient Model Weights Importance ---
            if self.model and self.model.weights is not None:
                try:
                    # Get feature names used in the model (assuming they match the training features)
                    model_features = [
                        'common_projects', 'win_rate', 'bid_pattern_similarity',
                        'dept_overlap_pct', 'price_range_overlap', 'recent_activity_score'
                    ]
                    # Check if scaler has feature names (StandardScaler doesn't store them by default)
                    # Use the predefined list 'model_features'

                    if len(model_features) == len(self.model.weights):
                         weights_df = pd.DataFrame({
                             'Feature': [f.replace('_', ' ').title() for f in model_features],
                             'Weight': self.model.weights,
                             'Absolute Weight': np.abs(self.model.weights)
                         }).sort_values('Absolute Weight', ascending=False)


                         plt.figure(figsize=(10, 6))
                         sns.barplot(x='Weight', y='Feature', data=weights_df, palette='vlag') # Use diverging palette
                         plt.title('Feature Weights from Gradient Descent Model')
                         plt.xlabel('Weight Value (Learned by Model)')
                         plt.tight_layout()
                         plt.savefig(f'{output_dir}/gradient_model_weights.png', dpi=300, bbox_inches='tight')
                         plt.close()
                         logger.info("Created gradient model weights visualization")
                    else:
                         logger.warning("Mismatch between number of model features and weights. Skipping weights plot.")

                except Exception as e_weights:
                    logger.error(f"Error creating model weights visualization: {e_weights}")
                    plt.close()


        except Exception as e:
            logger.error(f"Error creating feature importance visualization: {str(e)}")
            logger.exception(e)
            plt.close()


    def _create_rank_comparison(
        self,
        competitors_df: pd.DataFrame,
        known_competitors: List[str],
        methods: List[str],
        output_dir: str
    ) -> None:
        """Create visualization comparing ranks of known competitors across scoring methods."""
        try:
            if competitors_df.empty or not known_competitors or not methods:
                logger.warning("Insufficient data for rank comparison visualization.")
                return

            # Calculate ranks for each method
            rank_data = []
            competitor_names = {} # Map TIN to name

            for method in methods:
                 if method not in competitors_df.columns: continue # Skip if method doesn't exist

                 # Sort and rank (handle NaNs)
                 sorted_df = competitors_df.sort_values(method, ascending=False, na_position='last').reset_index(drop=True)
                 sorted_df[f'{method}_rank'] = sorted_df.index + 1

                 # Get ranks of known competitors
                 for tin in known_competitors:
                     competitor_row = sorted_df[sorted_df['tin'] == tin]

                     if not competitor_row.empty:
                         # Get name if not already fetched
                         if tin not in competitor_names:
                              name_row = competitors_df[competitors_df['tin'] == tin]
                              competitor_names[tin] = name_row['name'].iloc[0] if not name_row.empty and 'name' in name_row else tin # Fallback to TIN


                         competitor_name = competitor_names[tin]
                         competitor_rank = competitor_row[f'{method}_rank'].iloc[0]

                         rank_data.append({
                             'Competitor': competitor_name,
                             'Method': method.replace('_', ' ').title(),
                             'Rank': int(competitor_rank) if pd.notna(competitor_rank) else None # Handle potential NaN rank
                         })


            # Convert to DataFrame for visualization
            rank_df = pd.DataFrame(rank_data)
            rank_df = rank_df.dropna(subset=['Rank']) # Drop rows where rank is None



            if not rank_df.empty:
                 # --- Bar Chart ---
                 try:
                      num_competitors = rank_df['Competitor'].nunique()
                      plt.figure(figsize=(max(8, num_competitors * 1.5), 6)) # Dynamic width

                      # Create grouped bar chart
                      sns.barplot(x='Competitor', y='Rank', hue='Method', data=rank_df, palette='Set2') # Changed palette

                      # Invert y-axis so lower ranks (better) appear higher
                      plt.gca().invert_yaxis()

                      plt.title('Known Competitor Ranks by Scoring Method')
                      plt.xlabel('Competitor')
                      plt.ylabel('Rank (Lower is Better)')
                      plt.legend(title='Scoring Method', bbox_to_anchor=(1.05, 1), loc='upper left') # Adjust legend position
                      plt.xticks(rotation=45, ha='right')
                      plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend

                      plt.savefig(f'{output_dir}/rank_comparison_bars.png', dpi=300, bbox_inches='tight') # Changed filename
                      plt.close()

                      logger.info("Created rank comparison bar chart visualization")
                 except Exception as e_bar:
                      logger.error(f"Error creating rank comparison bar chart: {e_bar}")
                      plt.close()


                 # --- Line Plot ---
                 try:
                      num_methods = rank_df['Method'].nunique()
                      plt.figure(figsize=(max(8, num_methods * 1.5), 6)) # Dynamic width 

                      # Create line plot
                      sns.lineplot(x='Method', y='Rank', hue='Competitor', data=rank_df, marker='o', palette='tab10') # Changed palette

                      # Invert y-axis
                      plt.gca().invert_yaxis()

                      plt.title('Known Competitor Rank Trends Across Scoring Methods')
                      plt.xlabel('Scoring Method')
                      plt.ylabel('Rank (Lower is Better)')
                      plt.legend(title='Competitor', bbox_to_anchor=(1.05, 1), loc='upper left')
                      plt.grid(True, alpha=0.4, linestyle='--') # Adjusted grid
                      plt.xticks(rotation=45, ha='right') # Rotate x-axis labels if needed
                      plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend 

                      plt.savefig(f'{output_dir}/rank_comparison_trends.png', dpi=300, bbox_inches='tight') # Changed filename
                      plt.close()

                      logger.info("Created rank trends line plot visualization")
                 except Exception as e_line:
                      logger.error(f"Error creating rank comparison line plot: {e_line}")
                      plt.close()


        except Exception as e:
            logger.error(f"Error creating rank comparison visualization: {str(e)}")
            logger.exception(e)
            plt.close() # Ensure plot is closed on error


async def main():
    """Main entry point for the script."""
    start_time = time.time()
    logger.info("Starting Enhanced Competitor Scoring Validator")

    # Initialize database connection
    db = DbConnection()
    await db.initialize()

    try:
        # Create scoring validator
        validator = EnhancedScoringValidator(db)

        # Target company TIN (บริษัท เรืองฤทัย จำกัด - same as in our previous test)
        target_company_tin = "0105543041542"

        # Known strong competitors to validate against (used for training gradient model) 
        known_strong_competitors = [
            "0105540085581", # บริษัท ตรีสกุล จำกัด
            "0103503000975", # ห้างหุ้นส่วนจำกัด แสงนิยม
            "0105538044211", # บริษัท ปาล์ม คอน จำกัด
            # "0105561013814", # บริษัท พี.วาย.เอส.คอนสตรั๊คชั่น จำกัด - Example: maybe exclude one if needed for testing
        ]

        # Optional: Validation set (if you want to test generalization)
        # validation_set = ["0105561013814"]
        validation_set = None


        # Run the comprehensive competitor analysis
        results = await validator.analyze_competitors(
            target_company_tin=target_company_tin,
            known_competitors=known_strong_competitors,
            validation_set=validation_set # Pass validation set if defined
        )

        # --- Print Results ---
        if not results or 'target_company' not in results:
             print("\nAnalysis did not complete successfully. Check logs.")
             return # Exit if results are incomplete

        print("\n====== COMPETITOR SCORING ANALYSIS SUMMARY ======")
        print(f"Target Company: {results.get('target_company', {}).get('name', 'N/A')} ({target_company_tin})")
        print(f"Total Competitors Analyzed: {results.get('competitors_count', 'N/A')}")
        print(f"Known Strong Competitors (Training): {results.get('known_competitors_count', 'N/A')}")
        print(f"Analysis Completed in {results.get('execution_time', -1):.2f} seconds")

        # Print method evaluation results
        print("\n====== SCORING METHOD EVALUATION (Known Competitors) ======") # Clarified title

        if 'method_evaluation' in results and results['method_evaluation']:
             eval_metrics = results['method_evaluation'].get('metrics', {})
             if eval_metrics:
                 methods = list(eval_metrics.keys())
                 # Format header dynamically
                 header = f"{'Metric':<20} | " + ' | '.join([f"{m.replace('_', ' ').title():^15}" for m in methods])
                 print(header)
                 print("-" * len(header))


                 # Key metrics to display
                 metrics_to_show = [
                     'avg_rank', 'median_rank', 'min_rank', 'max_rank',
                     'top_10', 'top_20',
                     'avg_percentile', 'ndcg_10', 'precision_10'
                 ]

                 for metric in metrics_to_show:
                     values = []
                     for method in methods:
                          method_data = eval_metrics.get(method, {})
                          value = method_data.get(metric)
                          if isinstance(value, (int, float)):
                               # Format based on type
                               fmt = ".0f" if metric in ['top_10', 'top_20', 'min_rank', 'max_rank'] else ".2f"
                               values.append(f"{value:{fmt}}")
                          else:
                               values.append("N/A") # Handle missing metric data


                     print(f"{metric.replace('_', ' ').title():<20} | {' | '.join(f'{val:^15}' for val in values)}") 


             # Print method comparison
             method_comp = results['method_evaluation'].get('method_comparison', {})
             if method_comp:
                 print("\n====== METHOD COMPARISON (vs Base Method) ======")
                 for method, comparison in method_comp.items():
                     base_method_name = comparison.get('vs_method', 'N/A').replace('_', ' ').title()
                     print(f"\n{method.replace('_', ' ').title()} vs {base_method_name}:")
                     print(f"  Average Rank Improvement: {comparison.get('avg_rank_improvement', 0):+.2f}") # Added sign
                     print(f"  Percentile Improvement: {comparison.get('percentile_improvement', 0):+.2f}%")
                     print(f"  NDCG@10 Improvement:    {comparison.get('ndcg_10_improvement', 0):+.2f}")
                     print(f"  Precision@10 Improvement: {comparison.get('precision_10_improvement', 0):+.2f}")

        else:
             print("  No evaluation results available.")


        # Print target competitors analysis
        print("\n====== KNOWN COMPETITOR DETAILS ======") # Changed title
        if 'target_competitors' in results and results['target_competitors']:
             comp_details = results['target_competitors'].get('competitor_details', [])
             if comp_details:
                 for competitor in comp_details:
                     print(f"\nCompetitor: {competitor.get('name', 'N/A')} ({competitor.get('tin', 'N/A')})")
                     print(f"  Total Projects: {competitor.get('total_projects', 'N/A')}")
                     print(f"  Win Rate: {competitor.get('win_rate', 0):.2f}%")
                     print(f"  Common Projects w/ Target: {competitor.get('common_projects', 'N/A')}")

                     print("  Scores:")
                     scores = competitor.get('scores', {})
                     ranks = competitor.get('ranks', {}) # Get ranks too if available in evaluation results
                     if scores:
                          # Get ranks from evaluation if available
                          eval_comp_analysis = results.get('method_evaluation', {}).get('competitor_analysis', [])
                          comp_eval_data = next((item for item in eval_comp_analysis if item['tin'] == competitor['tin']), None)
                          ranks = comp_eval_data.get('ranks', {}) if comp_eval_data else {}


                          for method, score in scores.items():
                              rank = ranks.get(method, 'N/A')
                              rank_str = f"(Rank: {rank})" if rank != 'N/A' else ""
                              score_str = f"{score:.2f}" if score is not None else "N/A"
                              print(f"    {method.replace('_', ' ').title():<18}: {score_str:<8} {rank_str}")
                     else:
                          print("    No scores available.")

             else:
                  print("  No details found for known competitors.")


             # Print common characteristics
             common = results['target_competitors'].get('common_characteristics', {})
             if common:
                 print("\n====== COMMON CHARACTERISTICS OF THESE COMPETITORS ======") # Changed title
                 print(f"Average Win Rate: {common.get('avg_win_rate', 0):.2f}%")
                 print(f"Average Project Count: {common.get('avg_project_count', 0):.2f}")

                 if common.get('common_departments'):
                     print(f"Common Departments: {', '.join(common['common_departments'])}")

                 if common.get('common_project_types'):
                     print(f"Common Project Types: {', '.join(common['common_project_types'])}")


             # Print recommendations
             recs = results['target_competitors'].get('recommendations', {})
             if recs:
                 print("\n====== STRATEGIC INSIGHTS & RECOMMENDATIONS ======")

                 if recs.get('insights'):
                     print("\nInsights:")
                     for i, insight in enumerate(recs['insights'], 1):
                         print(f"  {i}.{insight}")

                 if recs.get('recommendations'):
                     print("\nRecommendations:")
                     for i, recommendation in enumerate(recs['recommendations'], 1):
                         print(f"  {i}. {recommendation}")

        else:
             print("  No target competitor analysis available.")


        # Print execution information
        results_dir = os.path.abspath(validator.output_dir)
        print(f"\nAnalysis complete. Detailed results saved to '{results_dir}' directory.")
        print(f"Log file saved to '{os.path.abspath('scoring_validation.log')}'")


    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.exception(e) # Log stack trace
        print(f"\nAn error occurred during execution. Please check the log file 'scoring_validation.log' for details.") # User-friendly message
    finally:
        # Close database connection
        if db and db.pool:
            await db.close()
        total_time = time.time() - start_time
        logger.info(f"Script execution completed in {total_time:.2f} seconds")


if __name__ == "__main__":
    # Ensure event loop compatibility for different OS
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())