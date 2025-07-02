"""
Data Persistence Service for Evolution and Training Results
Manages database operations for storing and retrieving training sessions
"""
import json
from datetime import datetime
from typing import Dict, List, Optional
from models import (
    TrainingSession, Generation, Chromosome, TrainingResult,
    init_db, get_db, close_db
)
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)

class PersistenceService:
    """Service for managing persistent storage of training data"""
    
    def __init__(self):
        """Initialize the persistence service"""
        try:
            init_db()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def create_training_session(self, name: str, config: Dict) -> str:
        """Create a new training session and return its ID"""
        db = get_db()
        try:
            session = TrainingSession(
                name=name,
                config=config,
                status='running'
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            
            session_id = str(session.id)
            logger.info(f"Created training session: {session_id}")
            return session_id
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create training session: {e}")
            raise
        finally:
            close_db(db)
    
    def save_generation_data(self, session_id: str, generation_data: Dict):
        """Save generation data including chromosomes"""
        db = get_db()
        try:
            # Create generation record
            generation = Generation(
                session_id=session_id,
                generation_number=generation_data.get('generation', 0),
                best_fitness=generation_data.get('maxFitness', 0.0),
                avg_fitness=generation_data.get('avgFitness', 0.0),
                max_fitness=generation_data.get('maxFitness', 0.0),
                diversity=generation_data.get('diversity', 0.0)
            )
            db.add(generation)
            db.commit()
            db.refresh(generation)
            
            # Save chromosomes
            chromosomes_data = generation_data.get('chromosomes', [])
            for idx, chrome_data in enumerate(chromosomes_data):
                chromosome = Chromosome(
                    generation_id=generation.id,
                    chromosome_index=idx,
                    fitness=chrome_data.get('fitness', 0.0),
                    is_champion=chrome_data.get('is_champion', False),
                    genes_data=chrome_data.get('genes', [])
                )
                db.add(chromosome)
            
            db.commit()
            logger.info(f"Saved generation {generation_data.get('generation')} with {len(chromosomes_data)} chromosomes")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save generation data: {e}")
            raise
        finally:
            close_db(db)
    
    def save_training_results(self, session_id: str, results: Dict):
        """Save final training results"""
        db = get_db()
        try:
            training_result = TrainingResult(
                session_id=session_id,
                episode_rewards=results.get('episode_rewards', []),
                final_performance=results.get('final_performance', {}),
                agent_stats=results.get('agent_stats', {}),
                best_features=results.get('best_features', []),
                feature_columns=results.get('feature_columns', [])
            )
            db.add(training_result)
            db.commit()
            
            logger.info(f"Saved training results for session {session_id}")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save training results: {e}")
            raise
        finally:
            close_db(db)
    
    def complete_training_session(self, session_id: str, best_fitness: float, final_generation: int):
        """Mark training session as completed"""
        db = get_db()
        try:
            session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
            if session:
                session.status = 'completed'
                session.end_time = datetime.utcnow()
                session.best_fitness = best_fitness
                session.final_generation = final_generation
                db.commit()
                
                logger.info(f"Completed training session {session_id}")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to complete training session: {e}")
            raise
        finally:
            close_db(db)
    
    def get_latest_session(self) -> Optional[Dict]:
        """Get the most recent training session with its data"""
        db = get_db()
        try:
            session = db.query(TrainingSession).order_by(TrainingSession.start_time.desc()).first()
            if not session:
                return None
            
            # Get latest generation
            latest_generation = (db.query(Generation)
                               .filter(Generation.session_id == session.id)
                               .order_by(Generation.generation_number.desc())
                               .first())
            
            if not latest_generation:
                return None
            
            # Get chromosomes for latest generation
            chromosomes = (db.query(Chromosome)
                         .filter(Chromosome.generation_id == latest_generation.id)
                         .order_by(Chromosome.chromosome_index)
                         .all())
            
            # Find champion
            champion = next((c for c in chromosomes if c.is_champion), None)
            
            # Format response
            result = {
                'session_id': str(session.id),
                'session_name': session.name,
                'status': session.status,
                'generation': latest_generation.generation_number,
                'max_generations': session.config.get('NUM_GENERATIONS', 30),
                'best_fitness': latest_generation.best_fitness,
                'is_active': session.status == 'running',
                'chromosomes': [
                    {
                        'id': idx,
                        'fitness': c.fitness,
                        'is_champion': c.is_champion,
                        'genes': c.genes_data
                    }
                    for idx, c in enumerate(chromosomes)
                ],
                'champion': {
                    'fitness': champion.fitness,
                    'genes': champion.genes_data
                } if champion else None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get latest session: {e}")
            return None
        finally:
            close_db(db)
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get complete evolution history for a session"""
        db = get_db()
        try:
            generations = (db.query(Generation)
                         .filter(Generation.session_id == session_id)
                         .order_by(Generation.generation_number)
                         .all())
            
            history = []
            for gen in generations:
                chromosomes = (db.query(Chromosome)
                             .filter(Chromosome.generation_id == gen.id)
                             .order_by(Chromosome.chromosome_index)
                             .all())
                
                history.append({
                    'generation': gen.generation_number,
                    'best_fitness': gen.best_fitness,
                    'avg_fitness': gen.avg_fitness,
                    'diversity': gen.diversity,
                    'timestamp': gen.timestamp.isoformat(),
                    'chromosomes': [
                        {
                            'fitness': c.fitness,
                            'is_champion': c.is_champion,
                            'genes': c.genes_data
                        }
                        for c in chromosomes
                    ]
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get session history: {e}")
            return []
        finally:
            close_db(db)
    
    def get_all_sessions(self) -> List[Dict]:
        """Get list of all training sessions"""
        db = get_db()
        try:
            sessions = (db.query(TrainingSession)
                       .order_by(TrainingSession.start_time.desc())
                       .all())
            
            return [
                {
                    'id': str(s.id),
                    'name': s.name,
                    'status': s.status,
                    'start_time': s.start_time.isoformat(),
                    'end_time': s.end_time.isoformat() if s.end_time else None,
                    'best_fitness': s.best_fitness,
                    'final_generation': s.final_generation,
                    'total_generations': s.config.get('NUM_GENERATIONS', 30)
                }
                for s in sessions
            ]
            
        except Exception as e:
            logger.error(f"Failed to get sessions: {e}")
            return []
        finally:
            close_db(db)
    
    def mark_all_sessions_completed(self):
        """Mark all active training sessions as completed"""
        db = get_db()
        try:
            # Update all sessions with status 'running' to 'completed'
            db.execute(text("UPDATE training_sessions SET status = 'completed' WHERE status = 'running'"))
            db.commit()
            logger.info("All active sessions marked as completed")
        except Exception as e:
            logger.error(f"Failed to mark sessions as completed: {e}")
            db.rollback()
            raise
        finally:
            close_db(db)

# Global persistence service instance
persistence_service = PersistenceService()