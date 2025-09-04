import pennylane as qml
from pennylane import numpy as np

class FeedbackDrivenQRC:
    """Feedback-driven Quantum Reservoir Computer, based on the work "Feedback-Driven Quantum Reservoir Computing for Time-Series Analysis" by Kobayashi et al.

    Główne metody:
    - fit(X, y=..., warmup=500, ltr=2000): zbiera stany rezerwuaru i uczy liniowych wag wyjścia
    - predict(X, return_states=False): oblicza predykcje dla podanego wejścia

    Parametry klasy ustawiono tak, by zachować domyślne wartości z oryginalnego skryptu.
    """

    def __init__(self, N=8, layers=7, ain=1e-3, afb=2.0, seed=42,
                 device_name="default.qubit", shots=None):
        # ----------------------------
        # 1) Params (defaults based on the work)
        # ----------------------------
        self.N = N
        self.layers = layers
        self.ain = ain # input scaling, 0.001 used in most examples in the paper
        self.afb = afb # feedback scaling, 2.5-3 in the paper mostly(needs to be optimized for tasks)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # urządzenie PennyLane (analityczne ⟨Z⟩)
        self.dev = qml.device(device_name, wires=self.N, shots=shots)

        # parametry rezerwuaru (losowane raz przy inicjalizacji)
        self.rots_kind, self.rots_theta = self.sample_reservoir_params(self.N, self.layers, self.rng)

        # QNode: zdefiniujemy go jako opakowanie funkcji obwodu
        self.qnode = qml.QNode(self._qrc_circuit, self.dev)

        # uczone/wyliczone po fit
        self.W_out_ = None
        self.fitted_ = False

    # ---------------------------------
    # 2) Definicja bramki R_{i,j}(theta)
    #    (Eq. (1) w artykule)
    # ---------------------------------
    @staticmethod
    def R_ij(i, j, theta):
        """
        Based on equation (1) in the paper: 
            Ri,j (θ ) = CXij RZj (θ )CXij RXi(θ )RXj (θ )
        but its different than the one in Fig. 2 (b):
            Ri,j (θ ) =  RXi(θ )RXj (θ ) CXij RZj (θ )CXij
        """
        qml.CNOT(wires=[i, j])
        qml.RZ(theta, wires=j)
        qml.CNOT(wires=[i, j])
        qml.RX(theta, wires=i)
        qml.RX(theta, wires=j)


        # 0: ─╭●───────────╭●──RX(0.00)─┤  
        # 1: ─╰X──RZ(0.00)─╰X──RX(0.00)─┤

    # -------------------------------------------------------
    # 3) Sprzętowo-efektywna wersja U_res (Appendix A, Fig.7)
    # -------------------------------------------------------
    def sample_reservoir_params(self, N, layers, rng):
        """
        For each layer: we randomly sample the rotation type (X/Y/Z) and angle for each qubit
        """
        rots_kind = []
        rots_theta = []
        for _ in range(layers):
            kinds = rng.integers(0, 3, size=N)     # 0->RX, 1->RY, 2->RZ
            thetas = rng.uniform(0.0, 2*np.pi, size=N)
            rots_kind.append(kinds)
            rots_theta.append(thetas)
        return np.array(rots_kind), np.array(rots_theta)

    def apply_reservoir(self, N, layers, rots_kind, rots_theta):
        """
        Hardware efficient version of U_res based on appendix A, Fig. 7 (a)

        """
        # wall of RY(pi/4)
        for w in range(N):
            qml.RY(np.pi/4, wires=w)
        # L layers of 1-qubit rotations in parallel + CNOT ladder
        for l in range(layers):
            for w in range(N):
                k = int(rots_kind[l, w])
                th = rots_theta[l, w]
                if k == 0:   qml.RX(th, wires=w)
                elif k == 1: qml.RY(th, wires=w)
                else:        qml.RZ(th, wires=w)
            # CNOT ladder: (0,1), (2,3), ... , then shifted (1,2), (3,4), ...
            for start in [0, 1]:
                for a in range(start, N-1, 2):
                    qml.CNOT(wires=[a, a+1])

            # Circuit for N = 8, Layers = 7 rots_kind=RX... and rots_theta= 180...
            # 0: ──RY(0.79)──RX(180.00)─╭●──RX(180.00)─────────────╭●──RX(180.00)─────────────╭●──RX(180.00)
            # 1: ──RY(0.79)──RX(180.00)─╰X─╭●───────────RX(180.00)─╰X─╭●───────────RX(180.00)─╰X─╭●─────────
            # 2: ──RY(0.79)──RX(180.00)─╭●─╰X───────────RX(180.00)─╭●─╰X───────────RX(180.00)─╭●─╰X─────────
            # 3: ──RY(0.79)──RX(180.00)─╰X─╭●───────────RX(180.00)─╰X─╭●───────────RX(180.00)─╰X─╭●─────────
            # 4: ──RY(0.79)──RX(180.00)─╭●─╰X───────────RX(180.00)─╭●─╰X───────────RX(180.00)─╭●─╰X─────────
            # 5: ──RY(0.79)──RX(180.00)─╰X─╭●───────────RX(180.00)─╰X─╭●───────────RX(180.00)─╰X─╭●─────────
            # 6: ──RY(0.79)──RX(180.00)─╭●─╰X───────────RX(180.00)─╭●─╰X───────────RX(180.00)─╭●─╰X─────────
            # 7: ──RY(0.79)──RX(180.00)─╰X──RX(180.00)─────────────╰X──RX(180.00)─────────────╰X──RX(180.00)

            # ──────────────╭●──RX(180.00)─────────────╭●──RX(180.00)─────────────╭●──RX(180.00)─────────────╭●────┤
            # ───RX(180.00)─╰X─╭●───────────RX(180.00)─╰X─╭●───────────RX(180.00)─╰X─╭●───────────RX(180.00)─╰X─╭●─┤
            # ───RX(180.00)─╭●─╰X───────────RX(180.00)─╭●─╰X───────────RX(180.00)─╭●─╰X───────────RX(180.00)─╭●─╰X─┤
            # ───RX(180.00)─╰X─╭●───────────RX(180.00)─╰X─╭●───────────RX(180.00)─╰X─╭●───────────RX(180.00)─╰X─╭●─┤
            # ───RX(180.00)─╭●─╰X───────────RX(180.00)─╭●─╰X───────────RX(180.00)─╭●─╰X───────────RX(180.00)─╭●─╰X─┤
            # ───RX(180.00)─╰X─╭●───────────RX(180.00)─╰X─╭●───────────RX(180.00)─╰X─╭●───────────RX(180.00)─╰X─╭●─┤
            # ───RX(180.00)─╭●─╰X───────────RX(180.00)─╭●─╰X───────────RX(180.00)─╭●─╰X───────────RX(180.00)─╭●─╰X─┤
            # ──────────────╰X──RX(180.00)─────────────╰X──RX(180.00)─────────────╰X──RX(180.00)─────────────╰X────┤

    # -------------------------------------------------------
    # 4) Mapowanie sprzężenia zwrotnego z z_{k-1} na pary (i,j)
    #    (na rys. 2 feedback idzie na kubity 3..8; tu wybieramy
    #    stałe, gęste mapowanie – dowolne stałe mapowanie jest OK)
    # -------------------------------------------------------
    # Dla przejrzystości: wejście kodujemy na (0,1); feedback na zestaw par
    feedback_pairs = [(2,3),(3,4),(4,5),(5,6),(6,7),(7,2),(0,2),(1,3)]  # 8 komponentów z_prev

    # -------------------------------------------------------
    # 5) Jeden cykl: s_k + z_{k-1} -> z_k
    # -------------------------------------------------------
    def _qrc_circuit(self, s_k, z_prev, ain, afb, rots_kind, rots_theta):
        # start w |0>^N

        # (i) wejście sk na kubity (0,1)
        self.R_ij(0, 1, ain * s_k)

        # (ii) feedback z_{k-1} rozlany po parach
        #     (wartości z_prev zakładamy w [-1,1], jak ⟨Z⟩)
        for alpha, (i,j) in enumerate(self.feedback_pairs):
            theta = afb * z_prev[alpha]
            self.R_ij(i, j, theta)

        # (iii) rezerwuar: splątanie i skramblowanie
        self.apply_reservoir(self.N, self.layers, rots_kind, rots_theta)

        # (iv) pomiar Z na wszystkich kubitach → wektor z_k
        return [qml.expval(qml.PauliZ(w)) for w in range(self.N)]

    # -------------------------------------------------------
    # 6) Pętla po sekwencji, trening i predykcja (wewnętrzne)
    # -------------------------------------------------------
    def _run_sequence(self, sequence, warmup=500, collect_all=True):
        """Uruchamia QRC po całej sekwencji i zwraca macierz Z_all (L x N).

        Jeśli collect_all==False zwróci tylko stany po warmup (przydatne do szybkiego testu)
        """
        L = len(sequence)
        z_prev = self.rng.uniform(0.0, 1.0, size=self.N)

        Z_all = np.zeros((L, self.N))
        for k in range(L):
            # qnode zwraca pennylane.numpy array - zamieniamy na float
            z_k = np.array(self.qnode(sequence[k], z_prev, self.ain, self.afb,
                                      self.rots_kind, self.rots_theta), dtype=float)
            Z_all[k] = z_k
            z_prev = z_k  # online feedback

        if collect_all:
            return Z_all
        else:
            return Z_all[warmup:]

    # Macierze projektorów do regresji liniowej (helper)
    @staticmethod
    def _make_X(Z_slice):
        ones = np.ones((Z_slice.shape[0], 1))
        return np.hstack([Z_slice, ones])

    # ----------------------------
    # 7) Metoda fit (zgodna ze sklearn)
    # ----------------------------
    def fit(self, sequence, y=None, warmup=500, ltr=2000, lts=None, regularize=1e-8):
        """Zbiera stany rezerwuaru dla całej sekwencji i dopasowuje wagę wyjścia.

        - sequence: 1D array sygnału wejściowego s_k
        - y: jeśli podane, to wektor celów o długości >= warmup + ltr
             jeśli nie podane, użytkownik musi sam zbudować y (np. STM opóźnienie)
        - lts: długość testu (opcjonalne). Nie jest wymagana do fit, ale
               pozwala na zachowanie kompatybilności z oryginalnym skryptem.
        """
        L = len(sequence)
        assert L >= warmup + ltr, "Sekwencja za krótka dla podanych warmup+ltr"

        Z_all = self._run_sequence(sequence, warmup=warmup, collect_all=True)

        # Wyciągamy część treningową
        X_tr = self._make_X(Z_all[warmup:warmup+ltr])

        if y is None:
            raise ValueError("Jeśli y==None, to nie mogę się nauczyć. Podaj cele y lub zbuduj zadanie STM przed fit.")

        y_tr = np.array(y[warmup:warmup+ltr], dtype=float)

        # Uczenie tylko wag wyjścia (Moore–Penrose) z małą regularizacją numeryczną
        # w = pinv(X) @ y  => z regularizacją użyjemy ridge-like rozwiązania: (X^T X + reg I)^{-1} X^T y
        XT_X = X_tr.T @ X_tr
        reg_mat = regularize * np.eye(XT_X.shape[0])
        self.W_out_ = np.linalg.solve(XT_X + reg_mat, X_tr.T @ y_tr)
        self.fitted_ = True

        return self

    # ----------------------------
    # 8) Metoda predict (zgodna ze sklearn)
    # ----------------------------
    def predict(self, sequence, warmup=500, start=None):
        """Dla podanej sekwencji oblicza stany rezerwuaru i zwraca predykcje.

        - jeśli start podany, predykcje będą od indeksu start (użyteczne gdy chcemy testować fragment)
        - domyślnie zakładamy, że sequence zaczyna w indeksie 0 i warmup jest stosowany tak jak przy fit
        """
        if not self.fitted_:
            raise RuntimeError("Model nie został dopasowany. Najpierw wywołaj fit().")

        Z_all = self._run_sequence(sequence, warmup=warmup, collect_all=True)

        # domyślnie bierzemy predykcje od warmup (chyba że użytkownik poda start)
        if start is None:
            start = warmup

        X = self._make_X(Z_all[start:])
        yhat = X @ self.W_out_
        return yhat


