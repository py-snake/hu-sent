from flask import Blueprint, render_template, request
from .models import Video, Comment, User, Tag, SentimentAnalysis
from . import db
from sqlalchemy import func

bp = Blueprint('main', __name__)


@bp.route('/')
def index():
    # Get filter and sort parameters from request
    sentiment_filter = request.args.get('sentiment', type=str)
    sort_by = request.args.get('sort_by', default='created_at', type=str)
    sort_order = request.args.get('sort_order', default='desc', type=str)

    # Get sentiment stats (unchanged)
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

    # Base query
    comments_query = Comment.query

    # Apply sentiment filter if specified
    if sentiment_filter in ['positive', 'negative', 'neutral']:
        comments_query = comments_query.join(SentimentAnalysis) \
            .filter(SentimentAnalysis.sentiment == sentiment_filter)

    # Apply sorting
    sort_field = {
        'date': Comment.created_at,
        'user': Comment.user_id,
        'upvotes': Comment.upvotes,
        'id': Comment.id
    }.get(sort_by, Comment.created_at)  # default to date

    if sort_order == 'asc':
        comments_query = comments_query.order_by(sort_field.asc())
    else:
        comments_query = comments_query.order_by(sort_field.desc())

    # Get comments (limit to 50 instead of 10 for better pagination)
    recent_comments = comments_query.limit(50).all()

    # Get counts (unchanged)
    video_count = Video.query.count()
    comment_count = Comment.query.count()
    user_count = User.query.count()

    return render_template('index.html',
                           recent_comments=recent_comments,
                           sentiment_data=sentiment_data,
                           video_count=video_count,
                           comment_count=comment_count,
                           user_count=user_count,
                           current_sentiment=sentiment_filter,
                           current_sort=sort_by,
                           current_order=sort_order)

@bp.route('/search')
def search():
    query = request.args.get('q', '')
    page = request.args.get('page', 1, type=int)
    search_type = request.args.get('type', 'comments')

    if search_type == 'comments':
        results = Comment.query.filter(Comment.text.ilike(f'%{query}%')) \
            .order_by(Comment.created_at.desc()) \
            .paginate(page=page, per_page=20)
    elif search_type == 'users':
        results = User.query.filter(User.username.ilike(f'%{query}%')) \
            .order_by(User.username.asc()) \
            .paginate(page=page, per_page=20)
    elif search_type == 'videos':
        results = Video.query.filter(Video.title.ilike(f'%{query}%')) \
            .order_by(Video.title.asc()) \
            .paginate(page=page, per_page=20)
    else:
        return 'Invalid search type', 400

    return render_template('search.html', results=results, query=query, type=search_type)

@bp.route('/video/<int:video_id>')
def video_comments(video_id):
    video = Video.query.filter_by(video_id=video_id).first_or_404()

    # Get filter and sort parameters from request
    sentiment_filter = request.args.get('sentiment', type=str)
    sort_by = request.args.get('sort_by', default='created_at', type=str)
    sort_order = request.args.get('sort_order', default='desc', type=str)

    # Base query
    comments_query = Comment.query.filter_by(video_id=video_id).join(SentimentAnalysis)

    # Apply sentiment filter if specified
    if sentiment_filter in ['positive', 'negative', 'neutral']:
        comments_query = comments_query.filter(SentimentAnalysis.sentiment == sentiment_filter)

    # Apply sorting
    sort_field = {
        'date': Comment.created_at,
        'upvotes': Comment.upvotes,
        'id': Comment.id
    }.get(sort_by, Comment.created_at)  # default to date

    if sort_order == 'asc':
        comments_query = comments_query.order_by(sort_field.asc())
    else:
        comments_query = comments_query.order_by(sort_field.desc())

    # Get comments
    comments = comments_query.all()

    # Get sentiment stats
    sentiment_stats = db.session.query(
        SentimentAnalysis.sentiment,
        func.count(SentimentAnalysis.id).label('count'),
        func.avg(SentimentAnalysis.confidence).label('avg_confidence')
    ).join(Comment).filter(Comment.video_id == video_id).group_by(SentimentAnalysis.sentiment).all()

    sentiment_data = {
        'total': sum(s.count for s in sentiment_stats) if sentiment_stats else 0,
        'stats': {s.sentiment: {
            'count': s.count,
            'percentage': round((s.count / sum(s.count for s in sentiment_stats)), 1) if sentiment_stats else 0,
            'avg_confidence': round(s.avg_confidence, 1) if s.avg_confidence else 0
        } for s in sentiment_stats} if sentiment_stats else {}
    }

    # Get counts
    total_comments = Comment.query.filter_by(video_id=video_id).count()

    return render_template('video.html',
                           video=video,
                           comments=comments,
                           sentiment_data=sentiment_data,
                           total_comments=total_comments,
                           current_sentiment=sentiment_filter,
                           current_sort=sort_by,
                           current_order=sort_order)


@bp.route('/user/<username>')
def user_comments(username):
    user = User.query.filter_by(username=username).first_or_404()

    # Get filter and sort parameters from request
    sentiment_filter = request.args.get('sentiment', type=str)
    sort_by = request.args.get('sort_by', default='created_at', type=str)
    sort_order = request.args.get('sort_order', default='desc', type=str)

    # Base query
    comments_query = Comment.query.filter_by(user_id=user.id).join(SentimentAnalysis)

    # Apply sentiment filter if specified
    if sentiment_filter in ['positive', 'negative', 'neutral']:
        comments_query = comments_query.filter(SentimentAnalysis.sentiment == sentiment_filter)

    # Apply sorting
    sort_field = {
        'date': Comment.created_at,
        'upvotes': Comment.upvotes,
        'id': Comment.id
    }.get(sort_by, Comment.created_at)  # default to date

    if sort_order == 'asc':
        comments_query = comments_query.order_by(sort_field.asc())
    else:
        comments_query = comments_query.order_by(sort_field.desc())

    # Get comments
    comments = comments_query.all()

    # Get sentiment stats
    sentiment_stats = db.session.query(
        SentimentAnalysis.sentiment,
        func.count(SentimentAnalysis.id).label('count'),
        func.avg(SentimentAnalysis.confidence).label('avg_confidence')
    ).join(Comment).filter(Comment.user_id == user.id).group_by(SentimentAnalysis.sentiment).all()

    sentiment_data = {
        'total': sum(s.count for s in sentiment_stats) if sentiment_stats else 0,
        'stats': {s.sentiment: {
            'count': s.count,
            'percentage': round((s.count / sum(s.count for s in sentiment_stats)), 1) if sentiment_stats else 0,
            'avg_confidence': round(s.avg_confidence, 1) if s.avg_confidence else 0
        } for s in sentiment_stats} if sentiment_stats else {}
    }

    # Get counts
    total_comments = Comment.query.filter_by(user_id=user.id).count()

    return render_template('user.html',
                           user=user,
                           comments=comments,
                           sentiment_data=sentiment_data,
                           total_comments=total_comments,
                           current_sentiment=sentiment_filter,
                           current_sort=sort_by,
                           current_order=sort_order)

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

