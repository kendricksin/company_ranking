#!/usr/bin/env python3
"""
Competitor Ranking Verification Script

This script reads the competitor analysis CSV file and verifies company rankings
across different scoring methods from the Enhanced Competitor Scoring Validator.
"""

import pandas as pd
import numpy as np
from tabulate import tabulate
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import matplotlib.font_manager as fm

# Set Thai font for matplotlib
font_paths = [
    'C:/Windows/Fonts/THSarabunNew.ttf',
    'C:/Windows/Fonts/THSarabunNew Bold.ttf',
    'C:/Windows/Fonts/Tahoma.ttf',  # Tahoma also supports Thai
    'C:/Windows/Fonts/TH Sarabun New.ttf',
    'C:/Windows/Fonts/THSarabun.ttf',
    'C:/Windows/Fonts/Arial.ttf',
    '/usr/share/fonts/truetype/thai-scalable/THSarabunNew.ttf',  # Linux path
    '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',  # Mac path
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Try to find available Thai font
thai_font = None
for font_path in font_paths:
    if os.path.exists(font_path):
        thai_font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = thai_font.get_name()
        logger.info(f"Using font: {thai_font.get_name()}")
        break

if not thai_font:
    logger.warning("No Thai font found, falling back to default font")

class RankingVerifier:
    """Class to verify competitor rankings from CSV data."""
    
    def __init__(self, csv_path):
        """Initialize with path to the CSV file."""
        self.csv_path = csv_path
        self.scoring_methods = [
            'current_score',
            'enhanced_score',
            'balanced_score',
            'weighted_score',
            'logistic_score',
            'gradient_score'
        ]
        
        # Known strong competitors from the original analysis
        self.known_strong_competitors = [
            "0105540085581",  # บริษัท ตรีสกุล จำกัด
            "0103503000975",  # ห้างหุ้นส่วนจำกัด แสงนิยม
            "0105538044211",  # บริษัท ปาล์ม คอน จำกัด
        ]
        
        # Target company from the original analysis
        self.target_company_tin = "0105543041542"  # บริษัท เรืองฤทัย จำกัด
        
        # Load the CSV data
        self.data = self._load_data()
        
    def _load_data(self):
        """Load and validate the CSV data."""
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Verify all required columns exist
            missing_cols = []
            for method in self.scoring_methods:
                if method not in df.columns:
                    missing_cols.append(method)
            
            if missing_cols:
                logger.warning(f"Missing scoring method columns: {missing_cols}")
            
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise
    
    def verify_rankings(self):
        """Generate ranking verification for all scoring methods."""
        print("\n" + "="*80)
        print("COMPETITOR RANKING VERIFICATION REPORT")
        print("="*80 + "\n")
        
        print(f"Total Companies Analyzed: {len(self.data)}")
        print(f"Known Strong Competitors: {len(self.known_strong_competitors)}")
        print()
        
        # Verify rankings for each scoring method
        for method in self.scoring_methods:
            self._verify_method_ranking(method)
        
        # Compare scoring methods
        self._compare_scoring_methods()
        
        # Analyze known strong competitors
        self._analyze_known_competitors()
        
        # Generate comprehensive ranking table for all companies
        self._generate_comprehensive_ranking_table()
        
        # Generate visualizations
        self._create_ranking_visualizations()
    
    def _verify_method_ranking(self, method):
        """Verify ranking for a specific scoring method."""
        if method not in self.data.columns:
            logger.warning(f"Scoring method '{method}' not found in CSV")
            return
        
        # Sort by the scoring method (descending)
        sorted_df = self.data.sort_values(method, ascending=False).reset_index(drop=True)
        sorted_df['rank'] = sorted_df.index + 1
        
        print(f"\n{'-'*60}")
        print(f"Ranking for: {method.replace('_', ' ').title()}")
        print(f"{'-'*60}")
        
        # Display top 10 competitors
        top_10 = sorted_df.head(10)[['rank', 'name', 'tin', method, 'common_projects', 'win_rate', 'total_projects']]
        print("\nTop 10 Competitors:")
        print(tabulate(top_10, headers='keys', tablefmt='grid', showindex=False, 
                      floatfmt=('.0f', '', '', '.2f', '.0f', '.2f', '.0f')))
        
        # Verify known strong competitors' ranks
        known_ranks = []
        for tin in self.known_strong_competitors:
            rank_row = sorted_df[sorted_df['tin'] == tin]
            if not rank_row.empty:
                rank = rank_row['rank'].iloc[0]
                name = rank_row['name'].iloc[0]
                score = rank_row[method].iloc[0]
                known_ranks.append((name, tin, rank, score))
        
        print("\nKnown Strong Competitors' Ranks:")
        for name, tin, rank, score in known_ranks:
            print(f"  {name} ({tin}): Rank {rank}, Score {score:.2f}")
        
        # Calculate key metrics
        avg_rank = np.mean([r[2] for r in known_ranks])
        median_rank = np.median([r[2] for r in known_ranks])
        top_10_count = sum(1 for r in known_ranks if r[2] <= 10)
        
        print(f"\nMetrics for {method}:")
        print(f"  Average Rank of Known Competitors: {avg_rank:.2f}")
        print(f"  Median Rank of Known Competitors: {median_rank:.2f}")
        print(f"  Known Competitors in Top 10: {top_10_count}")
    
    def _compare_scoring_methods(self):
        """Compare rankings across different scoring methods."""
        print("\n" + "="*60)
        print("SCORING METHOD COMPARISON")
        print("="*60 + "\n")
        
        # Create a comparison table for known strong competitors
        comparison_data = []
        
        for tin in self.known_strong_competitors:
            row_data = {'TIN': tin}
            
            competitor_row = self.data[self.data['tin'] == tin]
            if not competitor_row.empty:
                row_data['Name'] = competitor_row['name'].iloc[0]
                
                for method in self.scoring_methods:
                    if method in self.data.columns:
                        sorted_df = self.data.sort_values(method, ascending=False).reset_index(drop=True)
                        sorted_df['rank'] = sorted_df.index + 1
                        
                        rank = sorted_df[sorted_df['tin'] == tin]['rank'].iloc[0]
                        score = sorted_df[sorted_df['tin'] == tin][method].iloc[0]
                        
                        row_data[f'{method}_rank'] = rank
                        row_data[f'{method}_score'] = score
                
                comparison_data.append(row_data)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display ranking comparison
        rank_cols = ['Name'] + [f'{method}_rank' for method in self.scoring_methods if f'{method}_rank' in comparison_df.columns]
        print("Ranking Comparison for Known Strong Competitors:")
        rank_df = comparison_df[rank_cols].copy()
        rank_df.columns = ['Name'] + [m.replace('_score', '') for m in self.scoring_methods if f'{m}_rank' in comparison_df.columns]
        print(tabulate(rank_df, headers='keys', tablefmt='grid', showindex=False))
        
        # Calculate correlation between different scoring methods
        correlation_data = []
        for m1 in self.scoring_methods:
            if m1 in self.data.columns:
                row = {'Method': m1}
                for m2 in self.scoring_methods:
                    if m2 in self.data.columns:
                        corr = self.data[m1].corr(self.data[m2])
                        row[m2] = corr
                correlation_data.append(row)
        
        corr_df = pd.DataFrame(correlation_data)
        corr_df.set_index('Method', inplace=True)
        
        print("\nCorrelation between Scoring Methods:")
        print(corr_df.round(3).to_string())
    
    def _analyze_known_competitors(self):
        """Analyze performance of known strong competitors."""
        print("\n" + "="*60)
        print("KNOWN STRONG COMPETITORS ANALYSIS")
        print("="*60 + "\n")
        
        known_data = self.data[self.data['tin'].isin(self.known_strong_competitors)]
        
        if known_data.empty:
            print("No known competitor data found in CSV")
            return
        
        # Display detailed metrics for each known competitor
        for _, row in known_data.iterrows():
            print(f"\nCompetitor: {row['name']} ({row['tin']})")
            print(f"  Total Projects: {row['total_projects']}")
            print(f"  Total Wins: {row['total_wins']}")
            print(f"  Win Rate: {row['win_rate']:.2f}%")
            print(f"  Common Projects with Target: {row['common_projects']}")
            print(f"  Common Departments: {row['common_departments']}")
            print(f"  Recent Activity Score: {row['recent_activity_score']:.2f}")
            
            print("  Scores and Rankings:")
            for method in self.scoring_methods:
                if method in self.data.columns:
                    sorted_df = self.data.sort_values(method, ascending=False).reset_index(drop=True)
                    sorted_df['rank'] = sorted_df.index + 1
                    
                    rank = sorted_df[sorted_df['tin'] == row['tin']]['rank'].iloc[0]
                    score = row[method]
                    print(f"    {method.replace('_', ' ').title()}: Score {score:.2f}, Rank {rank}")
    
    def _generate_comprehensive_ranking_table(self):
        """Generate a comprehensive ranking table for all companies across all methods."""
        print("\n" + "="*60)
        print("COMPREHENSIVE RANKING FOR ALL 38 COMPANIES")
        print("="*60 + "\n")
        
        # Create a comprehensive ranking table
        ranking_data = []
        
        for _, company in self.data.iterrows():
            row_data = {
                'TIN': company['tin'],
                'Company': company['name'][:30] + '...' if len(company['name']) > 30 else company['name'],  # Truncate long names
                'Win Rate': f"{company['win_rate']:.1f}%",
                'Projects': company['total_projects']
            }
            
            # Calculate rank for each method
            for method in self.scoring_methods:
                if method in self.data.columns:
                    sorted_df = self.data.sort_values(method, ascending=False).reset_index(drop=True)
                    sorted_df['rank'] = sorted_df.index + 1
                    
                    rank = sorted_df[sorted_df['tin'] == company['tin']]['rank'].iloc[0]
                    score = company[method]
                    row_data[method] = f"{rank} ({score:.1f})"
            
            ranking_data.append(row_data)
        
        # Create DataFrame and save to CSV
        full_ranking_df = pd.DataFrame(ranking_data)
        
        # Sort by current_score rank (extract the numeric part)
        if 'current_score' in full_ranking_df.columns:
            full_ranking_df['current_rank'] = full_ranking_df['current_score'].str.extract(r'(\d+)').astype(int)
            full_ranking_df = full_ranking_df.sort_values('current_rank').drop('current_rank', axis=1)
        
        # Save to CSV
        full_ranking_df.to_csv('ranking_verification_output/comprehensive_rankings.csv', index=False, encoding='utf-8-sig')
        
        print("Comprehensive ranking table saved to 'ranking_verification_output/comprehensive_rankings.csv'")
        
        # Display summary statistics
        print("\nSummary Statistics for All Companies:")
        for method in self.scoring_methods:
            if method in self.data.columns:
                avg_score = self.data[method].mean()
                std_score = self.data[method].std()
                min_score = self.data[method].min()
                max_score = self.data[method].max()
                
                print(f"\n{method.replace('_', ' ').title()}:")
                print(f"  Average Score: {avg_score:.2f}")
                print(f"  Std Dev: {std_score:.2f}")
                print(f"  Min Score: {min_score:.2f}")
                print(f"  Max Score: {max_score:.2f}")

    def _create_ranking_visualizations(self):
        """Create visualizations for ranking verification."""
        try:
            # Create output directory for visualizations
            output_dir = 'ranking_verification_output'
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Rank comparison heatmap for top companies by each method
            self._create_top_companies_heatmap(output_dir)
            
            # 2. Rank distribution across methods (for all companies)
            self._create_rank_distribution_plot(output_dir)
            
            # 3. Score distribution by method
            plt.figure(figsize=(14, 8))
            data_for_plot = pd.DataFrame()
            
            for method in self.scoring_methods:
                if method in self.data.columns:
                    temp_df = pd.DataFrame({
                        'Score': self.data[method],
                        'Method': method.replace('_', ' ').title(),
                        'IsKnown': self.data['tin'].isin(self.known_strong_competitors)
                    })
                    data_for_plot = pd.concat([data_for_plot, temp_df], ignore_index=True)
            
            if not data_for_plot.empty:
                sns.boxplot(x='Method', y='Score', data=data_for_plot)
                # Overlay known competitors
                known_data = data_for_plot[data_for_plot['IsKnown']]
                sns.scatterplot(x='Method', y='Score', data=known_data, color='red', s=100, marker='o', 
                              label='Known Strong Competitors')
                
                plt.title('Score Distribution by Scoring Method')
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Score')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'{output_dir}/score_distribution_boxplot.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 4. Correlation matrix between scoring methods
            if len(self.scoring_methods) > 1:
                corr_matrix = self.data[self.scoring_methods].corr()
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
                plt.title('Correlation Matrix Between Scoring Methods')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/scoring_method_correlation.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Visualizations saved to '{output_dir}' directory")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
    
    def _create_top_companies_heatmap(self, output_dir):
        """Create a heatmap showing top 10 companies across different methods."""
        # Collect top 10 companies from each method
        top_companies_tins = set()
        for method in self.scoring_methods:
            if method in self.data.columns:
                sorted_df = self.data.sort_values(method, ascending=False).head(10)
                top_companies_tins.update(sorted_df['tin'].tolist())
        
        # Create comparison data for these companies
        comparison_data = []
        for tin in top_companies_tins:
            row_data = {'TIN': tin}
            competitor_row = self.data[self.data['tin'] == tin]
            
            if not competitor_row.empty:
                # Truncate name for display
                name = competitor_row['name'].iloc[0]
                row_data['Company'] = name[:25] + '...' if len(name) > 25 else name
                
                for method in self.scoring_methods:
                    if method in self.data.columns:
                        sorted_df = self.data.sort_values(method, ascending=False).reset_index(drop=True)
                        sorted_df['rank'] = sorted_df.index + 1
                        
                        rank = sorted_df[sorted_df['tin'] == tin]['rank'].iloc[0]
                        row_data[method] = rank
                
                comparison_data.append(row_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            # Sort by average rank across methods
            rank_columns = [col for col in comparison_df.columns if col not in ['TIN', 'Company']]
            comparison_df['avg_rank'] = comparison_df[rank_columns].mean(axis=1)
            comparison_df = comparison_df.sort_values('avg_rank').drop('avg_rank', axis=1)
            
            plt.figure(figsize=(14, 10))
            heat_data = comparison_df.set_index('Company')[self.scoring_methods].transpose()
            
            # Use English font for the heatmap labels
            sns.heatmap(heat_data, annot=True, cmap='YlOrRd_r', 
                       cbar_kws={'label': 'Rank (Lower is Better)'}, fmt='.0f',
                       xticklabels=[str(label) for label in heat_data.columns])
            
            plt.title('Ranking Comparison for Top Companies')
            plt.xlabel('Company (Truncated for Display)')
            plt.ylabel('Scoring Method')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/top_companies_rank_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_rank_distribution_plot(self, output_dir):
        """Create a distribution plot showing how ranks vary across methods."""
        plt.figure(figsize=(14, 8))
        
        # Calculate rank differences between methods
        if 'current_score' in self.data.columns:
            base_method = 'current_score'
            sorted_by_base = self.data.sort_values(base_method, ascending=False).reset_index(drop=True)
            sorted_by_base['base_rank'] = sorted_by_base.index + 1
            
            rank_differences = []
            
            for method in self.scoring_methods:
                if method != base_method and method in self.data.columns:
                    sorted_by_method = self.data.sort_values(method, ascending=False).reset_index(drop=True)
                    sorted_by_method['rank'] = sorted_by_method.index + 1
                    
                    # Calculate rank difference for each company
                    for _, row in self.data.iterrows():
                        base_rank = sorted_by_base[sorted_by_base['tin'] == row['tin']]['base_rank'].iloc[0]
                        method_rank = sorted_by_method[sorted_by_method['tin'] == row['tin']]['rank'].iloc[0]
                        
                        rank_differences.append({
                            'Method': method.replace('_', ' ').title(),
                            'Rank Difference': method_rank - base_rank,
                            'IsKnown': row['tin'] in self.known_strong_competitors
                        })
            
            rank_diff_df = pd.DataFrame(rank_differences)
            
            if not rank_diff_df.empty:
                sns.violinplot(x='Method', y='Rank Difference', data=rank_diff_df, inner='box')
                
                # Overlay known competitors
                known_data = rank_diff_df[rank_diff_df['IsKnown']]
                sns.swarmplot(x='Method', y='Rank Difference', data=known_data, color='red', size=10, 
                             label='Known Strong Competitors')
                
                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                plt.title(f'Rank Differences Compared to {base_method.replace("_", " ").title()}')
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Rank Difference (Positive = Ranked Lower)')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'{output_dir}/rank_difference_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()


def main():
    """Main function to run the ranking verification."""
    # Set up proper font handling for Thai characters
    if thai_font:
        plt.rcParams['font.sans-serif'] = [thai_font.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
    
    # Set the path to your CSV file
    csv_path = 'scoring_results/all_competitors_analysis.csv'
    
    # Check if file exists
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found at: {csv_path}")
        # Try alternative paths
        alternative_paths = [
            'all_competitors_analysis.csv',
            'scoring_results/scoring_results/all_competitors_analysis.csv',
            '../scoring_results/all_competitors_analysis.csv'
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                logger.info(f"Found CSV file at: {alt_path}")
                csv_path = alt_path
                break
        else:
            logger.error("Could not find the CSV file. Please check the path.")
            return
    
    # Create and run verifier
    verifier = RankingVerifier(csv_path)
    verifier.verify_rankings()


if __name__ == "__main__":
    main()