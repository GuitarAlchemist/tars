// Grammar Evolution 3D Scene
// Tier: 8 | Domains: ML, AI, NLP, Grammar | Performance: 92.0%

const grammarScene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });

// Tier visualization
const tierGeometry = new THREE.CylinderGeometry(0.5, 0.8, 2.40);
const tierMaterial = new THREE.MeshPhongMaterial({ color: 0x2ecc71, opacity: 0.94 });
const tierTower = new THREE.Mesh(tierGeometry, tierMaterial);
tierTower.position.set(0, 2.40 / 2, 0);
grammarScene.add(tierTower);

// Lighting
const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
directionalLight.position.set(10, 10, 5);
grammarScene.add(ambientLight);
grammarScene.add(directionalLight);

camera.position.set(5, 3, 5);
camera.lookAt(0, 0, 0);