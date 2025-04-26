from flask import Blueprint, render_template, request
from .models import Video, Comment, User, Tag

bp = Blueprint('main', __name__)


@bp.route('/')
def index():
    # Get some recent comments to display
    recent_comments = Comment.query.order_by(Comment.created_at.desc()).limit(10).all()
    video_count = Video.query.count()
    comment_count = Comment.query.count()
    user_count = User.query.count()

    return render_template('index.html',
                           recent_comments=recent_comments,
                           video_count=video_count,
                           comment_count=comment_count,
                           user_count=user_count)


@bp.route('/search')
def search():
    query = request.args.get('q', '')
    page = request.args.get('page', 1, type=int)

    if query:
        # Search in comments
        comments = Comment.query.filter(Comment.text.ilike(f'%{query}%')) \
            .order_by(Comment.created_at.desc()) \
            .paginate(page=page, per_page=20)
    else:
        # Show all comments if no query
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

