from . import db


class Video(db.Model):
    __tablename__ = 'videos'

    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, unique=True, nullable=False)
    title = db.Column(db.Text)
    description = db.Column(db.Text)
    url = db.Column(db.Text)
    download_link = db.Column(db.Text)

    comments = db.relationship('Comment', backref='video', lazy=True)
    tags = db.relationship('Tag', secondary='video_tags', backref='videos')


class Tag(db.Model):
    __tablename__ = 'tags'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Text, unique=True, nullable=False)


class VideoTag(db.Model):
    __tablename__ = 'video_tags'

    video_id = db.Column(db.Integer, db.ForeignKey('videos.video_id'), primary_key=True)
    tag_id = db.Column(db.Integer, db.ForeignKey('tags.id'), primary_key=True)


class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.Text, unique=True, nullable=False)
    profile_url = db.Column(db.Text)
    avatar_url = db.Column(db.Text)
    avatar_local_path = db.Column(db.Text)

    comments = db.relationship('Comment', backref='user', lazy=True)


class Comment(db.Model):
    __tablename__ = 'comments'

    id = db.Column(db.Integer, primary_key=True)
    comment_id = db.Column(db.Text, unique=True, nullable=False)
    video_id = db.Column(db.Integer, db.ForeignKey('videos.video_id'))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    parent_id = db.Column(db.Text, db.ForeignKey('comments.comment_id'))
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.Text)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    upvotes = db.Column(db.Integer, default=0)
    has_more_replies = db.Column(db.Boolean, default=False)
    reply_count = db.Column(db.Integer, default=0)

    replies = db.relationship('Comment', backref=db.backref('parent', remote_side=[comment_id]))
    mentions = db.relationship('Mention', backref='comment', lazy=True)


class Mention(db.Model):
    __tablename__ = 'mentions'

    id = db.Column(db.Integer, primary_key=True)
    comment_id = db.Column(db.Text, db.ForeignKey('comments.comment_id'))
    mentioned_username = db.Column(db.Text, db.ForeignKey('users.username'))

