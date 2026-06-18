const nav = document.querySelector('.topnav');

if (nav) {
  const updateNav = () => {
    nav.classList.toggle('scrolled', window.scrollY > 20);
  };
  updateNav();
  window.addEventListener('scroll', updateNav, { passive: true });
}
