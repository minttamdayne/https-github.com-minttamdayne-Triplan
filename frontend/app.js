const state = {
  itinerary: null,
  userInput: null,
  messages: [],
};

const $ = (id) => document.getElementById(id);

const els = {
  apiStatus: $("apiStatus"),
  form: $("tripForm"),
  planButton: $("planButton"),
  resetButton: $("resetButton"),
  resultTitle: $("resultTitle"),
  emptyState: $("emptyState"),
  loadingState: $("loadingState"),
  errorState: $("errorState"),
  summaryStats: $("summaryStats"),
  itineraryView: $("itineraryView"),
  chatPanel: $("chatPanel"),
  chatMessages: $("chatMessages"),
  chatForm: $("chatForm"),
  chatInput: $("chatInput"),
};

function todayOffset(days) {
  const date = new Date();
  date.setDate(date.getDate() + days);
  return date.toISOString().slice(0, 10);
}

function splitList(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function formatTime(value) {
  if (!value) return "--:--";
  return String(value).slice(0, 5);
}

function formatDate(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("vi-VN", {
    weekday: "long",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  }).format(new Date(`${value}T00:00:00`));
}

function moneyLabel(level) {
  return ["Miễn phí", "Tiết kiệm", "Vừa phải", "Thoải mái", "Cao cấp"][Number(level)] || "Vừa phải";
}

function showError(message) {
  els.errorState.textContent = message;
  els.errorState.classList.remove("hidden");
}

function clearError() {
  els.errorState.textContent = "";
  els.errorState.classList.add("hidden");
}

function setLoading(isLoading) {
  els.planButton.disabled = isLoading;
  els.planButton.querySelector("span").textContent = isLoading ? "Đang lập lịch trình..." : "Lập lịch trình";
  els.loadingState.classList.toggle("hidden", !isLoading);
}

function buildPayload() {
  const travelGroup = $("travelGroup").value;
  const mobility = $("limitedMobility").checked ? "limited" : "normal";
  const interests = splitList($("interests").value);

  return {
    interests,
    start_date: $("startDate").value,
    end_date: $("endDate").value,
    start_location: [Number($("latitude").value), Number($("longitude").value)],
    budget_level: Number($("budgetLevel").value),
    free_text: $("freeText").value.trim(),
    travel_group: travelGroup,
    group_type: travelGroup,
    pace: $("pace").value,
    mobility,
    has_children: $("hasChildren").checked,
    daily_hours: 10,
    max_places_per_day: 8,
    start_time: $("startTime").value,
    preferred_end_time: "21:00",
    nightlife_preference: $("hasChildren").checked ? "avoid" : "auto",
    food_priority: interests.some((item) => item.toLowerCase().includes("ẩm thực") || item.toLowerCase().includes("food"))
      ? "high"
      : "normal",
    culture_priority: "normal",
    outdoor_priority: "normal",
    food_preference: "normal",
    culture_preference: "normal",
    outdoor_preference: "normal",
    negative_preferences: [],
    food_restrictions: [],
    must_have: interests.slice(0, 2),
    avoid: [],
    themes: [],
    vibes: [],
    time_preferences: {},
    gender: "unknown",
  };
}

function renderStats(itinerary, userInput) {
  const days = itinerary.days || [];
  const stops = days.reduce((total, day) => total + (day.stops || []).length, 0);
  const travel = days.reduce((total, day) => total + Number(day.total_travel_minutes || 0), 0);
  const visit = days.reduce((total, day) => total + Number(day.total_visit_minutes || 0), 0);

  els.summaryStats.innerHTML = [
    ["Số ngày", days.length],
    ["Điểm dừng", stops],
    ["Di chuyển", `${Math.round(travel)} phút`],
    ["Ngân sách", moneyLabel(userInput.budget_level)],
  ]
    .map(([label, value]) => `<div class="stat"><strong>${value}</strong><span>${label}</span></div>`)
    .join("");
  els.summaryStats.classList.remove("hidden");

  return { stops, visit };
}

function renderItinerary(itinerary, userInput) {
  const days = itinerary.days || [];
  els.emptyState.classList.add("hidden");
  els.resultTitle.textContent = `${days.length || 0} ngày tại ${String(itinerary.city || "TP.HCM").toUpperCase()}`;
  renderStats(itinerary, userInput);

  els.itineraryView.innerHTML = days
    .map((day) => {
      const stops = day.stops || [];
      const stopMarkup = stops.length
        ? stops.map(renderStop).join("")
        : '<div class="stop-item"><div class="stop-time">--:--</div><div class="stop-body"><h4>Chưa có điểm dừng</h4><p>Planner không trả về địa điểm cho ngày này.</p></div></div>';

      return `
        <article class="day-card">
          <div class="day-head">
            <div>
              <h3>Ngày ${day.day_number}</h3>
              <p>${formatDate(day.date)}</p>
            </div>
            <p>${Math.round(Number(day.total_travel_minutes || 0) + Number(day.total_visit_minutes || 0))} phút tổng cộng</p>
          </div>
          <div class="stop-list">${stopMarkup}</div>
        </article>
      `;
    })
    .join("");

  els.chatPanel.classList.remove("hidden");
  els.chatMessages.innerHTML = "";
  state.messages = [
    {
      role: "assistant",
      content: "Mình đã có lịch trình rồi. Bạn có thể hỏi để đổi nhịp độ, thêm món ăn, giảm di chuyển hoặc tinh chỉnh từng ngày.",
    },
  ];
  renderMessages();
}

function renderStop(stop) {
  const poi = stop.poi || {};
  const tags = [
    poi.primaryType || poi.primary_type,
    poi.rating ? `${poi.rating} sao` : "",
    stop.visit_minutes ? `${stop.visit_minutes} phút` : "",
    ...(stop.matched_interests || []),
  ].filter(Boolean);

  return `
    <div class="stop-item">
      <div class="stop-time">${formatTime(stop.arrival_time)}</div>
      <div class="stop-body">
        <h4>${escapeHtml(poi.name || "Điểm dừng")}</h4>
        <p>${escapeHtml(poi.address || stop.reason || "Chưa có mô tả địa điểm.")}</p>
        ${stop.notes ? `<p>${escapeHtml(stop.notes)}</p>` : ""}
        ${stop.reason ? `<p>${escapeHtml(stop.reason)}</p>` : ""}
        ${tags.length ? `<div class="tag-row">${tags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}</div>` : ""}
        ${(stop.warnings || []).map((warning) => `<p class="warning">${escapeHtml(warning)}</p>`).join("")}
      </div>
    </div>
  `;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMessages() {
  els.chatMessages.innerHTML = state.messages
    .map((message) => `<div class="message ${message.role}">${escapeHtml(message.content)}</div>`)
    .join("");
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("API chưa sẵn sàng");
    els.apiStatus.textContent = "API sẵn sàng";
    els.apiStatus.classList.add("ok");
  } catch {
    els.apiStatus.textContent = "API lỗi";
    els.apiStatus.classList.add("bad");
  }
}

async function submitPlan(event) {
  event.preventDefault();
  clearError();
  setLoading(true);
  els.emptyState.classList.add("hidden");
  els.itineraryView.innerHTML = "";
  els.summaryStats.classList.add("hidden");
  els.chatPanel.classList.add("hidden");

  try {
    const payload = buildPayload();
    const response = await fetch("/api/v1/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Không thể lập lịch trình.");
    }
    state.itinerary = data;
    state.userInput = payload;
    renderItinerary(data, payload);
  } catch (error) {
    els.emptyState.classList.remove("hidden");
    showError(error.message || "Có lỗi khi gọi planner.");
  } finally {
    setLoading(false);
  }
}

async function submitChat(event) {
  event.preventDefault();
  const content = els.chatInput.value.trim();
  if (!content || !state.itinerary || !state.userInput) return;

  state.messages.push({ role: "user", content });
  els.chatInput.value = "";
  renderMessages();

  const pending = { role: "assistant", content: "Đang suy nghĩ..." };
  state.messages.push(pending);
  renderMessages();

  try {
    const response = await fetch("/api/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        itinerary: state.itinerary,
        user_input: state.userInput,
        messages: state.messages.filter((message) => message !== pending),
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Không thể chat với lịch trình.");
    pending.content = data.reply;
  } catch (error) {
    pending.content = error.message || "Chat đang gặp lỗi.";
  } finally {
    renderMessages();
  }
}

function resetView() {
  clearError();
  state.itinerary = null;
  state.userInput = null;
  state.messages = [];
  els.resultTitle.textContent = "Sẵn sàng lên kế hoạch";
  els.emptyState.classList.remove("hidden");
  els.summaryStats.classList.add("hidden");
  els.chatPanel.classList.add("hidden");
  els.itineraryView.innerHTML = "";
}

function initDates() {
  $("startDate").value = todayOffset(7);
  $("endDate").value = todayOffset(9);
}

initDates();
checkHealth();
els.form.addEventListener("submit", submitPlan);
els.chatForm.addEventListener("submit", submitChat);
els.resetButton.addEventListener("click", resetView);
