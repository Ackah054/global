self.addEventListener('install', function (event) {
  console.log('[ServiceWorker] Installed');
});

self.addEventListener('fetch', function (event) {
  event.respondWith(fetch(event.request));
});
