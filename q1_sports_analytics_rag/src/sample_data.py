from typing import List
from .models import DocumentUpload

class SampleDataGenerator:
    """Generates sample sports analytics data for testing."""
    
    @staticmethod
    def get_sample_documents() -> List[DocumentUpload]:
        """Get sample sports analytics documents."""
        return [
            DocumentUpload(
                content="""
                Premier League 2023-24 Season Analysis: Manchester City's Defensive Excellence

                Manchester City has established itself as the premier defensive unit in the 2023-24 Premier League season, conceding only 23 goals in 38 matches. This remarkable defensive record is the foundation of their title-winning campaign.

                Key Defensive Statistics:
                - Goals Conceded: 23 (lowest in the league)
                - Clean Sheets: 18 (highest in the league)
                - Expected Goals Against (xGA): 28.5
                - Tackles per game: 18.2
                - Interceptions per game: 12.8

                Goalkeeper Ederson has been instrumental in this defensive success, recording 15 clean sheets with a save percentage of 78.5%. His distribution accuracy of 94.2% has been crucial for City's build-up play.

                The defensive partnership of Ruben Dias and John Stones has been exceptional, with both players maintaining tackle success rates above 85%. Stones' ability to step into midfield has added a new dimension to City's defensive strategy.

                Manager Pep Guardiola's tactical innovations, including the use of inverted full-backs and a high defensive line, have created a system that prevents opposition attacks before they can develop into dangerous situations.
                """,
                metadata={
                    "type": "team_analysis",
                    "league": "Premier League",
                    "season": "2023-24",
                    "team": "Manchester City",
                    "category": "defensive_statistics"
                },
                source="premier_league_analysis_2024"
            ),
            
            DocumentUpload(
                content="""
                Lionel Messi's Goal-Scoring Performance Analysis: 2022-23 vs 2023-24 Seasons

                Lionel Messi's transition to Inter Miami has been marked by exceptional goal-scoring consistency, demonstrating his continued excellence in the twilight of his career.

                2022-23 Season (PSG):
                - Goals: 21 in 41 matches
                - Assists: 20 in 41 matches
                - Minutes per goal: 175.2
                - Shot conversion rate: 18.3%
                - Expected Goals (xG): 19.8

                2023-24 Season (Inter Miami):
                - Goals: 18 in 35 matches
                - Assists: 15 in 35 matches
                - Minutes per goal: 171.7
                - Shot conversion rate: 19.1%
                - Expected Goals (xG): 17.2

                Despite playing fewer matches in the 2023-24 season, Messi's efficiency has improved slightly, with a better shot conversion rate and marginally faster goal-scoring pace. His ability to create chances for teammates remains exceptional, with 15 assists demonstrating his continued playmaking prowess.

                The Argentine's performance in high-pressure situations has been particularly impressive, scoring 8 goals in the final 15 minutes of matches during the 2023-24 season, showcasing his clutch performance ability.
                """,
                metadata={
                    "type": "player_analysis",
                    "player": "Lionel Messi",
                    "seasons": "2022-23, 2023-24",
                    "teams": "PSG, Inter Miami",
                    "category": "goal_scoring_performance"
                },
                source="messi_performance_analysis_2024"
            ),
            
            DocumentUpload(
                content="""
                Goalkeeper Save Percentage Analysis: High-Pressure Situations in European Leagues

                Analysis of goalkeeper performance in high-pressure situations reveals significant variations in save percentages across different leagues and match contexts.

                Premier League Goalkeepers (2023-24):
                - Alisson (Liverpool): 84.2% overall, 78.5% in high-pressure situations
                - Ederson (Manchester City): 78.5% overall, 72.3% in high-pressure situations
                - David Raya (Arsenal): 76.8% overall, 71.2% in high-pressure situations

                La Liga Goalkeepers (2023-24):
                - Ter Stegen (Barcelona): 82.1% overall, 79.8% in high-pressure situations
                - Courtois (Real Madrid): 80.5% overall, 77.1% in high-pressure situations
                - Oblak (Atletico Madrid): 79.2% overall, 75.6% in high-pressure situations

                High-pressure situations are defined as:
                - Penalty kicks
                - One-on-one situations
                - Set pieces in the final 10 minutes
                - Matches where the team is leading by 1 goal or less

                Ter Stegen emerges as the most reliable goalkeeper in high-pressure situations, maintaining a save percentage of 79.8% compared to his overall rate of 82.1%. His performance in penalty shootouts has been particularly impressive, saving 4 out of 6 penalties faced.

                The data suggests that experience and mental fortitude play crucial roles in high-pressure performance, with veteran goalkeepers generally outperforming their younger counterparts in critical moments.
                """,
                metadata={
                    "type": "goalkeeper_analysis",
                    "leagues": "Premier League, La Liga",
                    "season": "2023-24",
                    "category": "save_percentage_analysis"
                },
                source="goalkeeper_performance_analysis_2024"
            ),
            
            DocumentUpload(
                content="""
                Top 3 Defensive Teams in European Football: 2023-24 Season Comparison

                A comprehensive analysis of defensive performance across Europe's top leagues reveals the most effective defensive units of the 2023-24 season.

                1. Manchester City (Premier League)
                - Goals Conceded: 23
                - Clean Sheets: 18
                - Expected Goals Against: 28.5
                - Tackles per game: 18.2
                - Interceptions per game: 12.8
                - Defensive efficiency rating: 9.2/10

                2. Inter Milan (Serie A)
                - Goals Conceded: 26
                - Clean Sheets: 16
                - Expected Goals Against: 30.1
                - Tackles per game: 19.5
                - Interceptions per game: 14.2
                - Defensive efficiency rating: 8.9/10

                3. Real Madrid (La Liga)
                - Goals Conceded: 28
                - Clean Sheets: 15
                - Expected Goals Against: 32.8
                - Tackles per game: 17.8
                - Interceptions per game: 13.1
                - Defensive efficiency rating: 8.7/10

                Key Defensive Statistics Comparison:
                Manchester City's defensive unit, led by Ruben Dias and John Stones, has been the most effective in preventing goals while maintaining possession-based football. Their ability to control games through possession (average 68.5%) reduces defensive workload while maintaining high defensive standards.

                Inter Milan's defensive approach focuses on aggressive pressing and tactical discipline, with manager Simone Inzaghi implementing a 3-5-2 system that provides defensive solidity while allowing attacking flexibility.

                Real Madrid's defensive success is built on the foundation of experienced defenders like Antonio Rudiger and David Alaba, combined with the tactical flexibility of manager Carlo Ancelotti's system.
                """,
                metadata={
                    "type": "comparative_analysis",
                    "leagues": "Premier League, Serie A, La Liga",
                    "season": "2023-24",
                    "teams": "Manchester City, Inter Milan, Real Madrid",
                    "category": "defensive_team_comparison"
                },
                source="top_defensive_teams_analysis_2024"
            ),
            
            DocumentUpload(
                content="""
                Champions League 2023-24: Knockout Stage Performance Analysis

                The 2023-24 Champions League knockout stages have provided compelling evidence of the evolving tactical landscape in European football, with several teams demonstrating exceptional defensive and attacking capabilities.

                Quarter-Final Results and Analysis:
                - Manchester City vs Real Madrid: 4-4 aggregate (City won on penalties)
                - Bayern Munich vs Arsenal: 3-2 aggregate
                - PSG vs Barcelona: 6-4 aggregate
                - Atletico Madrid vs Borussia Dortmund: 5-4 aggregate

                Semi-Final Performance Metrics:
                Manchester City's progression to the semi-finals was built on their defensive solidity, conceding only 2 goals in 4 knockout matches. Their possession-based approach, averaging 67.3% possession, has been effective in controlling games and limiting opposition opportunities.

                Bayern Munich's attacking prowess has been their defining characteristic, scoring 12 goals in 4 knockout matches. Harry Kane's contribution of 6 goals and 3 assists has been instrumental in their success.

                PSG's defensive improvements under Luis Enrique have been notable, with the team conceding only 4 goals in 4 matches while maintaining their attacking threat through Kylian Mbappé's 5 goals and 2 assists.

                Key Tactical Trends:
                - High defensive lines with aggressive pressing
                - Increased use of inverted full-backs
                - Emphasis on set-piece efficiency
                - Tactical flexibility in response to opposition strengths

                The data suggests that teams with balanced defensive and attacking capabilities, combined with tactical flexibility, are most successful in knockout competitions.
                """,
                metadata={
                    "type": "competition_analysis",
                    "competition": "Champions League",
                    "season": "2023-24",
                    "stage": "knockout",
                    "teams": "Manchester City, Bayern Munich, PSG, Real Madrid",
                    "category": "champions_league_analysis"
                },
                source="champions_league_knockout_analysis_2024"
            ),
            
            DocumentUpload(
                content="""
                Erling Haaland vs Kylian Mbappé: Goal-Scoring Efficiency Comparison 2023-24

                The ongoing rivalry between Erling Haaland and Kylian Mbappé continues to captivate football fans worldwide, with both players demonstrating exceptional goal-scoring abilities in the 2023-24 season.

                Erling Haaland (Manchester City):
                - Goals: 28 in 35 matches
                - Assists: 8 in 35 matches
                - Minutes per goal: 108.5
                - Shot conversion rate: 24.7%
                - Expected Goals (xG): 26.3
                - Big chances conversion: 78.3%

                Kylian Mbappé (PSG):
                - Goals: 32 in 38 matches
                - Assists: 12 in 38 matches
                - Minutes per goal: 102.3
                - Shot conversion rate: 22.1%
                - Expected Goals (xG): 29.8
                - Big chances conversion: 75.6%

                Performance Analysis:
                Haaland's efficiency in front of goal is remarkable, with a shot conversion rate of 24.7% and an xG overperformance of +1.7 goals. His ability to convert big chances at 78.3% demonstrates his clinical finishing ability.

                Mbappé's versatility is evident in his higher assist count (12 vs 8) and his ability to create chances for teammates. His slightly lower shot conversion rate is offset by his higher goal output and creative contributions.

                Contextual Performance:
                Haaland's performance in high-pressure matches (Champions League knockout stages, Premier League title race) has been exceptional, scoring 8 goals in 6 such matches.

                Mbappé's consistency across different competitions is notable, with goals in Ligue 1, Champions League, and international matches demonstrating his adaptability to different tactical systems and opposition quality.

                The comparison reveals two different but equally effective approaches to goal-scoring: Haaland's clinical efficiency versus Mbappé's dynamic versatility.
                """,
                metadata={
                    "type": "player_comparison",
                    "players": "Erling Haaland, Kylian Mbappé",
                    "season": "2023-24",
                    "teams": "Manchester City, PSG",
                    "category": "goal_scoring_comparison"
                },
                source="haaland_mbappe_comparison_2024"
            )
        ] 