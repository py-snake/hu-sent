from flask import Blueprint, render_template, request
from .models import Video, Comment, User, Tag, SentimentAnalysis
from . import db
from sqlalchemy import func

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    sentiment_stats = db.session.query(
        SentimentAnalysis.sentiment,
        func.count(SentimentAnalysis.id).label('count'),
        func.avg(SentimentAnalysis.confidence).label('avg_confidence')
    ).group_by(SentimentAnalysis.sentiment).all()

    sentiment_data = {
        'total': sum(s.count for s in sentiment_stats) if sentiment_stats else 0,
        'stats': {s.sentiment: {
            'count': s.count,
            'percentage': round((s.count / sum(s.count for s in sentiment_stats)), 1) if sentiment_stats else 0,
            'avg_confidence': round(s.avg_confidence, 1) if s.avg_confidence else 0
        } for s in sentiment_stats} if sentiment_stats else {}
    }

    recent_comments = Comment.query.order_by(Comment.created_at.desc()).limit(10).all()
    video_count = Video.query.count()
    comment_count = Comment.query.count()
    user_count = User.query.count()

    return render_template('index.html',
                         recent_comments=recent_comments,
                         sentiment_data=sentiment_data,
                         video_count=video_count,
                         comment_count=comment_count,
                         user_count=user_count)

@bp.route('/search')
def search():
    query = request.args.get('q', '')
    page = request.args.get('page', 1, type=int)

    if query:
        comments = Comment.query.filter(Comment.text.ilike(f'%{query}%')) \
            .order_by(Comment.created_at.desc()) \
            .paginate(page=page, per_page=20)
    else:
        comments = Comment.query.order_by(Comment.created_at.desc()) \
            .paginate(page=page, per_page=20)

    return render_template('search.html', comments=comments, query=query)

@bp.route('/video/<int:video_id>')
def video_comments(video_id):
    video = Video.query.filter_by(video_id=video_id).first_or_404()
    comments = Comment.query.filter_by(video_id=video_id, parent_id=None) \
        .order_by(Comment.created_at.desc()) \
        .all()
    return render_template('video.html', video=video, comments=comments)

@bp.route('/user/<username>')
def user_comments(username):
    user = User.query.filter_by(username=username).first_or_404()
    comments = Comment.query.filter_by(user_id=user.id) \
        .order_by(Comment.created_at.desc()) \
        .all()
    return render_template('user.html', user=user, comments=comments)


@bp.route('/stats')
def stats():
    # Get the same stats as the homepage
    sentiment_stats = db.session.query(
        SentimentAnalysis.sentiment,
        func.count(SentimentAnalysis.id).label('count'),
        func.avg(SentimentAnalysis.confidence).label('avg_confidence')
    ).group_by(SentimentAnalysis.sentiment).all()

    # Format the data exactly like the homepage
    sentiment_data = {
        'total': sum(s.count for s in sentiment_stats) if sentiment_stats else 0,
        'stats': {s.sentiment: {
            'count': s.count,
            'percentage': round((s.count / sum(s.count for s in sentiment_stats)), 1) if sentiment_stats else 0,
            'avg_confidence': round(s.avg_confidence, 1) if s.avg_confidence else 0
        } for s in sentiment_stats} if sentiment_stats else {}
    }

    # Get the same counts as homepage
    video_count = Video.query.count()
    comment_count = Comment.query.count()
    user_count = User.query.count()

    return render_template('stats.html',
                         sentiment_data=sentiment_data,
                         video_count=video_count,
                         comment_count=comment_count,
                         user_count=user_count)

