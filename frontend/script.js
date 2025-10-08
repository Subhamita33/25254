const container = document.querySelector('.container');
const SignUpBtn = document.querySelector('.SignUp-btn');
const loginBtn = document.querySelector('.login-btn');

SignUpBtn.addEventListener('click', () => {
    container.classList.add('active');
});

loginBtn.addEventListener('click', () => {
    container.classList.remove('active');
});
