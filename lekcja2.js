let gl_xaxis = [];
let gl_yaxis = [];
let gl_m, gl_b;
const gl_lerning_rate = 0.5;
//sgd = sarcastic gradient descent
const gl_optimizer = tf.train.sgd(gl_lerning_rate);

function setup() {
  createCanvas(1500,1500);
  background(0);
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
}

function mousePressed() {
  //mapowanie maping
  let l_x = map(mouseX, 0 , width, 0, 1);
  let l_y = map(mouseY, 0 ,height, 1, 0);
  gl_xaxis.push(l_x);
  gl_yaxis.push(l_y);
}

function draw() {
  background(0);
  stroke(255);
  strokeWeight(8);
  //rysowanie ponktow
  for (var i = 0; i < gl_xaxis.length; i++) {
    let l_point_x = map(gl_xaxis[i], 0, 1, 0, width);
    let l_point_y = map(gl_yaxis[i], 0, 1, height, 0);
    point(l_point_x, l_point_y)
  }
  tf.tidy(() => {
    if (gl_xaxis.length > 0) {
      const l_tf_yaxis = tf.tensor1d(gl_yaxis)
      function kjn_train() {
        return kjn_loss_function(kjn_predict(gl_xaxis), l_tf_yaxis);
      }
      gl_optimizer.minimize(kjn_train);

      // rysowanie lini
      // startowe punkty
      const l_xaxis = [0,1];
      const l_yaxis = kjn_predict(l_xaxis);
      let l_x1 = map(l_xaxis[0], 0, 1, 0, width);
      let l_x2 = map(l_xaxis[1], 0, 1, 0, width);

      let l_line_y = l_yaxis.dataSync();
      let l_y1 = map(l_line_y[0], 0, 1, height, 0);
      let l_y2 = map(l_line_y[1], 0, 1, height, 0);
      strokeWeight(4);
      line(l_x1, l_y1, l_x2, l_y2);
    }
  });
}

//funkcja która przewiduje y z x
function kjn_predict(p_xaxis) {
  const l_tf_xaxis = tf.tensor1d(p_xaxis);
  //y = m*a + b
  const l_tf_yaxis = l_tf_xaxis.mul(m).add(b);
  return l_tf_yaxis;
}

function kjn_loss_function(p_prediction, p_label) {
  //sqr(yp-y)
  return p_prediction.sub(p_label).square().mean();
}

//rysowanie lini
const gl_y_line = kjn_predict(gl_xaxis);
let gl_x1 = map(gl_xaxis[0], 0, 1, 0, width);
let gl_x2 = map(gl_xaxis[1], 0, 1, 0, width);

let gl_line_y = gl_yaxis.dataSync();
let gl_y1 = map(gl_line_y[0], 0, 1, height, 0);
let gl_y2 = map(gl_line_y[1], 0, 1, height, 0);
