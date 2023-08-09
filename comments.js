// Create web server using express
// Add a route for comments
// Add a route for comments/:id
// Add a route for comments/new
// Add a route for comments/:id/edit
// Add a route for comments/:id/delete
// Add a route for comments/:id
// Add a route for comments/:id/edit
// Add a route for comments/:id/delete

var express = require('express');
var app = express();
var bodyParser = require('body-parser');
var port = 3000;

var comments = [
  {id: 1, title: 'First Comment', body: 'This is the first comment'},
  {id: 2, title: 'Second Comment', body: 'This is the second comment'},
  {id: 3, title: 'Third Comment', body: 'This is the third comment'},
  {id: 4, title: 'Fourth Comment', body: 'This is the fourth comment'},
  {id: 5, title: 'Fifth Comment', body: 'This is the fifth comment'}
];

app.use(bodyParser.urlencoded({extended: true}));

app.set('view engine', 'ejs');

app.get('/', function(req, res) {
  res.render('index');
});

// Index route
app.get('/comments', function(req, res) {
  res.render('comments/index', {comments: comments});
});

// New route
app.get('/comments/new', function(req, res) {
  res.render('comments/new');
});

// Create route
app.post('/comments', function(req, res) {
  var newComment = {
    id: comments.length + 1,
    title: req.body.title,
    body: req.body.body
  };
  comments.push(newComment);
  res.redirect('/comments');
});

// Show route
app.get('/comments/:id', function(req, res) {
  var comment = comments.find(function(comment) {
    return comment.id === parseInt(req.params.id);
  });
  res.render('comments/show', {comment: comment});
});

// Edit route
app.get('/comments/:id/edit', function(req, res) {
  var comment = comments.find(function(comment) {
    return comment.id === parseInt(req.params.id);
  });
  res.render('comments/edit', {comment: comment});
});

// Update route
app.put('/comments/:id', function(req, res) {
  var comment = comments.find(function(comment) {
    return comment.id === parseInt(req.params.id); 
    res.redirect('/comments');
});
});
 